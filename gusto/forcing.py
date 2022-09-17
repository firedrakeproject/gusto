from firedrake import (Function, TrialFunctions, DirichletBC,
                       LinearVariationalProblem, LinearVariationalSolver)
from gusto.configuration import logger, DEBUG
from gusto.labels import (transport, diffusion, name, time_derivative,
                          replace_subject, hydrostatic)
from gusto.fml.form_manipulation_labelling import drop


__all__ = ["Forcing"]


class Forcing(object):
    """
    Base class for forcing terms for Gusto.

    """

    def __init__(self, equation, alpha):

        self.field_name = equation.field_name
        implicit_terms = ["incompressibility", "sponge"]
        dt = equation.state.dt

        W = equation.function_space
        self.x0 = Function(W)
        self.xF = Function(W)

        # set up boundary conditions on the u subspace of W
        bcs = [DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain) for bc in equation.bcs['u']]

        # drop terms relating to transport and diffusion
        residual = equation.residual.label_map(
            lambda t: any(t.has_label(transport, diffusion, return_tuple=True)), drop)

        # the lhs of both of the explicit and implicit solvers is just
        # the time derivative form
        trials = TrialFunctions(W)
        a = residual.label_map(lambda t: t.has_label(time_derivative),
                               replace_subject(trials),
                               map_if_false=drop)

        # the explicit forms are multiplied by (1-alpha) and moved to the rhs
        L_explicit = -(1-alpha)*dt*residual.label_map(
            lambda t: t.has_label(time_derivative) or t.get(name) in implicit_terms or t.get(name) == "hydrostatic_form",
            drop,
            replace_subject(self.x0))

        # the implicit forms are multiplied by alpha and moved to the rhs
        L_implicit = -alpha*dt*residual.label_map(
            lambda t: t.has_label(time_derivative) or t.get(name) in implicit_terms or t.get(name) == "hydrostatic_form",
            drop,
            replace_subject(self.x0))

        # now add the terms that are always fully implicit
        if any(t.get(name) in implicit_terms for t in residual):
            L_implicit -= dt*residual.label_map(
                lambda t: t.get(name) in implicit_terms,
                replace_subject(self.x0),
                drop)

        # the hydrostatic equations require some additional forms:
        if any([t.has_label(hydrostatic) for t in residual]):

            L_explicit += residual.label_map(
                lambda t: t.get(name) == "hydrostatic_form",
                replace_subject(self.x0),
                drop)

            L_implicit -= residual.label_map(
                lambda t: t.get(name) == "hydrostatic_form",
                replace_subject(self.x0),
                drop)

        # now we can set up the explicit and implicit problems
        explicit_forcing_problem = LinearVariationalProblem(
            a.form, L_explicit.form, self.xF, bcs=bcs
        )

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
