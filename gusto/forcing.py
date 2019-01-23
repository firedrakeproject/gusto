from firedrake import (TrialFunctions, Function,
                       DirichletBC, LinearVariationalProblem,
                       LinearVariationalSolver, Constant)
from gusto.configuration import logger, DEBUG
from gusto.form_manipulation_labelling import (drop, time_derivative,
                                               advection,
                                               replace_labelled)

__all__ = ["Forcing"]


class Forcing(object):
    """
    Base class for forcing terms for Gusto.

    """

    def __init__(self, equation, dt, alpha):

        # this is the function that the forcing term is applied to
        W = equation.function_space
        self.x0 = Function(W)
        self.xF = Function(W)

        eqn = equation().label_map(lambda t: t.has_label(advection), drop)
        assert len(eqn) > 1
        self._build_forcing_solver(W, eqn, dt, alpha)

    def _build_forcing_solver(self, W, equation, dt, alpha):

        trials = TrialFunctions(W)

        a = equation.label_map(lambda t: t.has_label(time_derivative),
                               replace_labelled("subject", trials),
                               drop)
        L_explicit = Constant(-(1-alpha)*dt)*equation.label_map(
            lambda t: not t.has_label(time_derivative),
            replace_labelled("subject", self.x0.split()),
            drop)
        L_implicit = Constant(-alpha*dt)*equation.label_map(
            lambda t: not t.has_label(time_derivative),
            replace_labelled("subject", self.x0.split()),
            drop)

        Vu = W.split()[0]
        if Vu.extruded:
            bcs = [DirichletBC(Vu, 0.0, "bottom"),
                   DirichletBC(Vu, 0.0, "top")]
        else:
            bcs = None

        explicit_forcing_problem = LinearVariationalProblem(
            a.form, L_explicit.form, self.xF, bcs=bcs
        )

        implicit_forcing_problem = LinearVariationalProblem(
            a.form, L_implicit.form, self.xF, bcs=bcs
        )

        solver_parameters = {}
        if logger.isEnabledFor(DEBUG):
            solver_parameters["ksp_monitor_true_residual"] = True

        self.solvers = {}
        self.solvers["explicit"] = LinearVariationalSolver(
            explicit_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix="ExplicitForcingSolver"
        )
        self.solvers["implicit"] = LinearVariationalSolver(
            implicit_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix="ImplicitForcingSolver"
        )

    def apply(self, x_in, x_nl, x_out, label):
        """
        Function takes x_in as input, computes F(x_nl) and returns
        x_out = x_in + F(x_nl)
        as output.

        :arg x_in: :class:`.Function` object
        :arg x_nl: :class:`.Function` object
        :arg x_out: :class:`.Function` object
        """

        self.x0.assign(x_nl)
        x_out.assign(x_in)

        self.solvers[label].solve()
        x = x_out
        x += self.xF
