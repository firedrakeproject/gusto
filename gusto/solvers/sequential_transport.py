""" Code for conservative transport, for splitting the nonlinear 
problem into two linear problems which are solved sequentially: first the 
density and then for the mixing ratio.  """

from firedrake import (
    Function,
    LinearVariationalProblem,
    LinearVariationalSolver,
    TrialFunction,
    action,
    derivative,
    split,
)
from ufl import replace, zero
from firedrake import assemble, action
from firedrake.petsc import PETSc


# Use the location from which TimeDiscretisation already imports this.
from firedrake.formmanipulation import split_form

def affine_forms(residual, solution):
    """
    Convert an affine residual R(u; v) = 0 into a linear problem

        a(u, v) = L(v).

    If R(u; v) = A(u, v) - b(v), then

        a = A,
        L = b.
    """
    trial = TrialFunction(solution.function_space())

    # A(du, v)
    a = derivative(residual, solution, trial)

    # For an affine residual:
    #
    #     action(a, solution) - residual
    #       = A(solution, v) - [A(solution, v) - b(v)]
    #       = b(v).
    L = action(a, solution) - residual

    return a, L

def assembled_vector_norm(form):
    """Return the Euclidean norm of an assembled linear residual form."""
    residual = assemble(form)

    with residual.dat.vec_ro as vec:
        return vec.norm()

def constant_test_residual(form, function_space):
    """
    Evaluate a scalar residual using the constant test function v = 1.
    """
    one = Function(function_space, name="constant_test")
    one.assign(1.0)

    return float(assemble(action(form, one)))


class SequentialConservativeTransportSolver:
    """
    Solve a triangular mixed conservative-transport residual.

    The mixed state is assumed to be ordered as

        (rho, m_1, ..., m_N),

    or the corresponding indices must be supplied explicitly.

    Publicly, this object behaves like a Firedrake variational solver:
    it exposes a zero-argument solve() method.
    """

    def __init__(
        self,
        residual,
        solution,
        density_index=0,
        tracer_indices=(1,),
        bcs=None,
        options_prefix="",
        density_parameters=None,
        tracer_parameters=None,
    ):
        self.residual = residual
        self.solution = solution
        self.W = solution.function_space()

        self.density_index = density_index
        self.tracer_indices = tuple(tracer_indices)
        self.options_prefix = options_prefix

        nfields = len(solution.subfunctions)

        if density_index < 0 or density_index >= nfields:
            raise ValueError(
                f"Density index {density_index} is invalid for a mixed "
                f"space containing {nfields} fields."
            )

        for idx in self.tracer_indices:
            if idx < 0 or idx >= nfields:
                raise ValueError(
                    f"Tracer index {idx} is invalid for a mixed space "
                    f"containing {nfields} fields."
                )

        if density_index in self.tracer_indices:
            raise ValueError(
                "The density field cannot also be a tracer field."
            )

        # Scalar DG transport normally has no essential boundary conditions
        # in the closed/periodic cases in the paper. Mixed BCs require an
        # additional step to assign each BC to the appropriate block.
        if bcs:
            raise NotImplementedError(
                "Sequential conservative transport does not yet split "
                "mixed Dirichlet boundary conditions. The scalar transport "
                "tests considered here should not require essential BCs."
            )

        default_parameters = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": None,
        }

        self.density_parameters = (
            default_parameters.copy()
            if density_parameters is None
            else density_parameters.copy()
        )

        self.tracer_parameters = (
            default_parameters.copy()
            if tracer_parameters is None
            else tracer_parameters.copy()
        )

        # Expressions for the components of x_out occurring in the original
        # mixed residual.
        self.mixed_components = tuple(split(solution))

        # Standalone block Functions. These become the unknowns in the
        # individual linear variational problems.
        self.work = tuple(
            Function(
                subfunction.function_space(),
                name=f"sequential_transport_field_{i}",
            )
            for i, subfunction in enumerate(solution.subfunctions)
        )

        # Replace all components of the original mixed unknown by the
        # standalone working Functions.
        self.component_replacements = {
            mixed_component: work_component
            for mixed_component, work_component in zip(
                self.mixed_components, self.work
            )
        }

        self.residual_blocks = self._extract_residual_blocks()

        self.density_solver = self._build_density_solver()
        self.tracer_solvers = self._build_tracer_solvers()

        print('Sequential linear transport solver initialised')

    def _extract_residual_blocks(self):
            """
            Split the full residual by test-function block and replace the mixed
            output components by standalone block Functions.
            """
            split_residual = split_form(self.residual)

            if len(split_residual) != len(self.work):
                raise ValueError(
                    "The number of residual equation blocks does not match the "
                    "number of fields in the mixed solution: "
                    f"{len(split_residual)} blocks versus {len(self.work)} fields."
                )

            blocks = []

            for block in split_residual:
                # In the current Gusto usage, split_form returns objects with a
                # .form attribute. Accommodate a plain UFL form as well.
                block_form = block.form if hasattr(block, "form") else block

                blocks.append(
                    replace(block_form, self.component_replacements)
                )

            return tuple(blocks)

    def _build_density_solver(self):
        rho = self.work[self.density_index]
        F_rho = self.residual_blocks[self.density_index]

        a_rho, L_rho = affine_forms(F_rho, rho)

        problem = LinearVariationalProblem(
            a_rho,
            L_rho,
            rho,
        )

        return LinearVariationalSolver(
            problem,
            solver_parameters=self.density_parameters,
            options_prefix=(
                self.options_prefix + "_sequential_density"
            ),
        )

    def _build_tracer_solvers(self):
        solvers = []

        for tracer_index in self.tracer_indices:
            tracer = self.work[tracer_index]
            F_tracer = self.residual_blocks[tracer_index]

            # This derivative treats every other work Function, including
            # the density, as a known coefficient.
            a_tracer, L_tracer = affine_forms(
                F_tracer,
                tracer,
            )

            problem = LinearVariationalProblem(
                a_tracer,
                L_tracer,
                tracer,
            )

            solver = LinearVariationalSolver(
                problem,
                solver_parameters=self.tracer_parameters,
                options_prefix=(
                    self.options_prefix
                    + f"_sequential_tracer_{tracer_index}"
                ),
            )

            solvers.append((tracer_index, solver))

        return tuple(solvers)

    def solve(self):
        """
        Solve the original triangular mixed residual.
        """
        # Gusto has already placed the previous stage value in the output
        # Function as an initial guess. Transfer it to the standalone fields.
        for work_field, mixed_field in zip(
                self.work, self.solution.subfunctions):
            work_field.assign(mixed_field)

        # First solve the density equation.
        self.density_solver.solve()

        # Then solve each tracer equation. The updated density work Function
        # is a coefficient in each tracer form.
        for _, tracer_solver in self.tracer_solvers:
            tracer_solver.solve()

        # Return the block solutions to Gusto's mixed stage Function.
        self.solution.subfunctions[self.density_index].assign(
            self.work[self.density_index]
        )

        for tracer_index, _ in self.tracer_solvers:
            self.solution.subfunctions[tracer_index].assign(
                self.work[tracer_index]
            )


    