from firedrake import LinearVariationalProblem, LinearVariationalSolver
from gusto.state import FieldCreator
from gusto.transport_terms import TransportTerm
from gusto.configuration import DEBUG


__all__ = ["Forcing"]


class Forcing(object):
    """
    Class defining forcing solvers for split timestepping schemes.

    :arg state: x :class:`.State` object.
    :arg equations: x :class:`.Equations` object.
    """

    def __init__(self, state, equations):
        self.state = state

        self.x0 = FieldCreator(equations)

        # this is the function that contains the result of solving
        # <test, trial> = <test, F(x0)>, where F is the forcing term
        self.xF = FieldCreator()
        self.solvers = {"explicit": {}, "implicit": {}}

        for field in equations.fieldlist:
            if not all([isinstance(term, TransportTerm) for name, term in equations(field).terms.items()]):
                self.xF(field, state.fields(field).function_space())
                self._build_forcing_solver(field, equations(field))

    def _build_forcing_solver(self, field, equation):
        a = equation.mass_term(equation.trial)
        L_explicit = 0.
        L_implicit = 0.
        dt = self.state.timestepping.dt
        for name, term in equation.terms.items():
            if not isinstance(term, TransportTerm):
                L_explicit += dt * term.off_centering * term(self.x0(field), self.x0)
                L_implicit += dt * (1. - term.off_centering) * term(self.x0(field), self.x0)

        explicit_forcing_problem = LinearVariationalProblem(
            a, L_explicit, self.xF(field), bcs=equation.bcs
        )

        implicit_forcing_problem = LinearVariationalProblem(
            a, L_implicit, self.xF(field), bcs=equation.bcs
        )

        solver_parameters = {}
        if self.state.output.log_level == DEBUG:
            solver_parameters["ksp_monitor_true_residual"] = True

        self.solvers["explicit"][field] = LinearVariationalSolver(
            explicit_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix=field+"ExplicitForcingSolver"
        )
        self.solvers["implicit"][field] = LinearVariationalSolver(
            implicit_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix=field+"ImplicitForcingSolver"
        )

    def apply(self, x_in, x_nl, x_out, label):
        """
        Function takes x_in as input, computes F(x_nl) and returns
        x_out = x_in + F(x_nl)
        as output.

        :arg x_in: :class:`.Function` object
        :arg x_nl: :class:`.Function` object
        :arg x_out: :class:`.Function` object
        :arg label: :str: specifying whether the forcing is explicit
        or implicit. For the explicit solves, each forcing term is multiplied
        by its off_centering value, theta, as specified in the term class.
        For the implicit solves, this factor is replaced by (1 - theta).

        """
        self.x0('xfields').assign(x_nl)
        x_out('xfields').assign(x_in)

        for field, solver in self.solvers[label].items():
            solver.solve()
            x = x_out(field)
            x += self.xF(field)
