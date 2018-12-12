from firedrake import (TestFunctions, TrialFunctions, Function,
                       DirichletBC, LinearVariationalProblem,
                       LinearVariationalSolver, Constant)
from gusto.configuration import DEBUG
from gusto.form_manipulation_labelling import Term, drop, time_derivative, advection, all_terms
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

    def __init__(self, state, equation):

        fieldlist = equation.fieldlist

        # this is the function that the forcing term is applied to
        self.x0 = Function(state.spaces("W"))
        self.xF = Function(state.spaces("W"))

        eqn = equation().label_map(lambda t: t.has_label(advection), drop)

        tests = TestFunctions(state.spaces("W"))

        def replace_test(t):
            field = t.labels["prognostic_variable"]
            test = t.form.arguments()[0]
            form = ufl.replace(t.form, {test: tests[fieldlist.index(field)]})
            return Term(form, t.labels)

        eqn = eqn.label_map(all_terms, replace_test)
        assert len(eqn) > 1
        self._build_forcing_solver(state, fieldlist, eqn)

    def _build_forcing_solver(self, state, fieldlist, equation):

        W = state.spaces.W
        trials = TrialFunctions(W)
        alpha = state.timestepping.alpha
        dt = state.timestepping.dt

        def replace_subject_with_trial(t):
            field = t.labels["subject"]
            trial = trials[fieldlist.index(field.name())]
            form = ufl.replace(t.form, {t.labels["subject"]: trial})
            return Term(form, t.labels)

        def replace_subject(label):
            if label == "explicit":
                const = Constant((1-alpha)*dt)
            elif label == "implicit":
                const = Constant(alpha*dt)

            def replacer(t):
                field = t.labels["subject"]
                x = self.x0.split()[fieldlist.index(field.name())]
                form = const*ufl.replace(t.form,
                                         {t.labels["subject"]: x})
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
        if state.output.log_level == DEBUG:
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
