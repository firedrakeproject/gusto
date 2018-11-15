from firedrake import (TrialFunction,
                       DirichletBC, LinearVariationalProblem,
                       LinearVariationalSolver, Constant)
from gusto.configuration import DEBUG
from gusto.form_manipulation_labelling import Term, drop, time_derivative, advection
from gusto.state import FieldCreator
import ufl


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

    def __init__(self, state, equations):

        self.equations = equations
        fieldlist = [f for f, _ in equations]

        # this is the function that the forcing term is applied to
        self.x0 = FieldCreator(fieldlist, state.spaces("W"))
        self.xF = FieldCreator(fieldlist, state.spaces("W"))

        self.solvers = {"explicit": {}, "implicit": {}}
        for field, equation in equations:
            eqn = equation.label_map(lambda t: t.has_label(advection),
                                     drop)
            if len(eqn) > 1:
                self._build_forcing_solver(state, field, eqn)

    def _build_forcing_solver(self, state, field, equation):

        V = state.fields(field).function_space()
        trial = TrialFunction(V)
        alpha = state.timestepping.alpha
        dt = state.timestepping.dt

        def replace_subject_with_trial(t):
            form = ufl.replace(t.form, {t.labels["subject"]: trial})
            return Term(form, t.labels)

        def replace_subject(label):
            if label == "explicit":
                const = Constant((1-alpha)*dt)
            elif label == "implicit":
                const = Constant(alpha*dt)

            def replacer(t):
                form = const*ufl.replace(t.form,
                                         {t.labels["subject"]:
                                          self.x0(t.labels["subject"].name())})
                return Term(form, t.labels)
            return replacer

        a = equation.label_map(lambda t: t.has_label(time_derivative),
                               replace_subject_with_trial,
                               drop)
        L_explicit = equation.label_map(
            lambda t: not t.has_label(time_derivative),
            replace_subject("explicit"),
            drop)
        L_implicit = equation.label_map(
            lambda t: not t.has_label(time_derivative),
            replace_subject("implicit"),
            drop)

        if V.extruded:
            bcs = [DirichletBC(V, 0.0, "bottom"),
                   DirichletBC(V, 0.0, "top")]
        else:
            bcs = None

        explicit_forcing_problem = LinearVariationalProblem(
            a.form, L_explicit.form, self.xF(field), bcs=bcs
        )

        implicit_forcing_problem = LinearVariationalProblem(
            a.form, L_implicit.form, self.xF(field), bcs=bcs
        )

        solver_parameters = {}
        if state.output.log_level == DEBUG:
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
        """

        self.x0.X.assign(x_nl)
        x_out.assign(x_in)

        for field, solver in self.solvers[label].items():
            solver.solve()
        x = x_out
        x += self.xF.X
