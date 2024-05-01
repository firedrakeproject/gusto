"""Discretisation of dynamic forcing terms, such as the pressure gradient."""

from firedrake import (
    Function, TrialFunctions, DirichletBC, LinearVariationalProblem,
    LinearVariationalSolver
)
from firedrake.fml import drop, replace_subject
from gusto.labels import (
    transport, diffusion, time_derivative, hydrostatic, physics_label,
    sponge, incompressible
)
from gusto.logging import logger, DEBUG, logging_ksp_monitor_true_residual


__all__ = ["Forcing"]


class Forcing(object):
    """
    Discretises forcing terms.

    This class describes the evaluation of forcing terms, for instance the
    gravitational force, the Coriolis force or the pressure gradient force.
    These are terms that can simply be evaluated, generally as part of some
    semi-implicit time discretisation.
    """

    def __init__(self, equation, alpha):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the prognostic equations
                containing the forcing terms.
            alpha (:class:`Constant`): semi-implicit off-centering factor. An
                alpha of 0 corresponds to fully explicit, while a factor of 1
                corresponds to fully implicit.
        """

        self.field_name = equation.field_name
        implicit_terms = [incompressible, sponge]
        dt = equation.domain.dt

        W = equation.function_space
        self.x0 = Function(W)
        self.xF = Function(W)

        # set up boundary conditions on the u subspace of W
        bcs = [DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain) for bc in equation.bcs['u']]

        # drop terms relating to transport, diffusion and physics
        residual = equation.residual.label_map(
            lambda t: any(t.has_label(transport, diffusion, physics_label,
                                      return_tuple=True)), drop)

        # the lhs of both of the explicit and implicit solvers is just
        # the time derivative form
        trials = TrialFunctions(W)
        a = residual.label_map(lambda t: t.has_label(time_derivative),
                               replace_subject(trials),
                               map_if_false=drop)

        # the explicit forms are multiplied by (1-alpha) and moved to the rhs
        L_explicit = -(1-alpha)*dt*residual.label_map(
            lambda t:
                any(t.has_label(time_derivative, hydrostatic, *implicit_terms,
                                return_tuple=True)),
            drop,
            replace_subject(self.x0))

        # the implicit forms are multiplied by alpha and moved to the rhs
        L_implicit = -alpha*dt*residual.label_map(
            lambda t:
                any(t.has_label(
                    time_derivative, hydrostatic, *implicit_terms,
                    return_tuple=True)),
            drop,
            replace_subject(self.x0))

        # now add the terms that are always fully implicit
        L_implicit -= dt*residual.label_map(
            lambda t: any(t.has_label(*implicit_terms, return_tuple=True)),
            replace_subject(self.x0),
            drop)

        # the hydrostatic equations require some additional forms:
        if any([t.has_label(hydrostatic) for t in residual]):
            L_explicit += residual.label_map(
                lambda t: t.has_label(hydrostatic),
                replace_subject(self.x0),
                drop)

            L_implicit -= residual.label_map(
                lambda t: t.has_label(hydrostatic),
                replace_subject(self.x0),
                drop)

        # now we can set up the explicit and implicit problems
        explicit_forcing_problem = LinearVariationalProblem(
            a.form, L_explicit.form, self.xF, bcs=bcs
        )

        implicit_forcing_problem = LinearVariationalProblem(
            a.form, L_implicit.form, self.xF, bcs=bcs
        )

        self.solvers = {}
        self.solvers["explicit"] = LinearVariationalSolver(
            explicit_forcing_problem,
            options_prefix="ExplicitForcingSolver"
        )
        self.solvers["implicit"] = LinearVariationalSolver(
            implicit_forcing_problem,
            options_prefix="ImplicitForcingSolver"
        )

        if logger.isEnabledFor(DEBUG):
            self.solvers["explicit"].snes.ksp.setMonitor(logging_ksp_monitor_true_residual)
            self.solvers["implicit"].snes.ksp.setMonitor(logging_ksp_monitor_true_residual)

    def apply(self, x_in, x_nl, x_out, label):
        """
        Applies the discretisation for a forcing term F(x).

        This takes x_in and x_nl and computes F(x_nl), and updates x_out to   \n
        x_out = x_in + scale*F(x_nl)                                          \n
        where 'scale' is the appropriate semi-implicit factor.

        Args:
            x_in (:class:`FieldCreator`): the field to be incremented.
            x_nl (:class:`FieldCreator`): the field which the forcing term is
                evaluated on.
            x_out (:class:`FieldCreator`): the output field to be updated.
            label (str): denotes which forcing to apply. Should be 'explicit' or
                'implicit'. TODO: there should be a check on this. Or this
                should be an actual label.
        """

        self.x0.assign(x_nl(self.field_name))

        self.solvers[label].solve()  # places forcing in self.xF

        x_out.assign(x_in(self.field_name))
        x_out += self.xF
