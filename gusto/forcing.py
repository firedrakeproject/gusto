from firedrake import (Function, TrialFunctions, DirichletBC,
                       LinearVariationalProblem, LinearVariationalSolver)
from gusto.configuration import logger, DEBUG
from gusto.labels import (transport, diffusion, name, time_derivative,
                          replace_subject)
from gusto.fml.form_manipulation_labelling import drop


__all__ = ["Forcing"]


class Forcing(object):
    """
    Base class for forcing terms for Gusto.

    :arg state: x :class:`.State` object.
    :arg euler_poincare: if True then the momentum equation is in Euler
    Poincare form and we need to add 0.5*grad(u^2) to the forcing term.
    If False then this term is not added.
    :arg linear: if True then we are solving a linear equation so nonlinear
    terms (namely the Euler Poincare term) should not be added.
    :arg extra_terms: extra terms to add to the u component of the forcing
    term - these will be multiplied by the appropriate test function.
    """

    def __init__(self, equation, dt, alpha):

        self.field_name = equation.field_name
        implicit_terms = ["divergence_form", "sponge"]

        W = equation.function_space
        self.x0 = Function(W)
        self.xF = Function(W)

        residual = equation.residual.label_map(
            lambda t: any(t.has_label(transport, diffusion, return_tuple=True)), drop)

        trials = TrialFunctions(W)
        a = residual.label_map(lambda t: t.has_label(time_derivative),
                               replace_subject(trials),
                               map_if_false=drop)

        L_explicit = -(1-alpha)*dt*residual.label_map(
            lambda t: t.has_label(time_derivative) or t.get(name) in implicit_terms,
            drop,
            replace_subject(self.x0))

        bcs = [DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain) for bc in equation.bcs['u']]

        explicit_forcing_problem = LinearVariationalProblem(
            a.form, L_explicit.form, self.xF, bcs=bcs
        )

        L_implicit = -alpha*dt*residual.label_map(
            lambda t: t.has_label(time_derivative) or t.get(name) in implicit_terms,
            drop,
            replace_subject(self.x0))
        if any(t.get(name) in implicit_terms for t in residual):
            L_implicit -= dt*residual.label_map(
                lambda t: t.get(name) in implicit_terms,
                replace_subject(self.x0),
                drop)

        implicit_forcing_problem = LinearVariationalProblem(
            a.form, L_implicit.form, self.xF, bcs=bcs
        )

        solver_parameters = {}
        if logger.isEnabledFor(DEBUG):
            solver_parameters["ksp_monitor_true_residual"] = None

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
        Function takes x as input, computes F(x_nl) and returns
        x_out = x + scale*F(x_nl)
        as output.

        :arg x_in: :class:`.Function` object
        :arg x_nl: :class:`.Function` object
        :arg x_out: :class:`.Function` object
        """

        self.x0.assign(x_nl(self.field_name))

        self.solvers[label].solve()  # places forcing in self.xF

        x_out.assign(x_in(self.field_name))
        x_out += self.xF
