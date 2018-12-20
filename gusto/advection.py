from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import (Function, LinearVariationalProblem,
                       LinearVariationalSolver, Projector, Interpolator,
                       TestFunction, TrialFunction, FunctionSpace,
                       BrokenElement, Constant, dot, grad)
from firedrake.utils import cached_property
from gusto.form_manipulation_labelling import (all_terms, advection,
                                               time_derivative, drop,
                                               replace_test, replace_labelled,
                                               extract)
from gusto.recovery import Recoverer


__all__ = ["ForwardEuler", "SSPRK3", "ThetaMethod"]


def embedded_dg(original_apply):
    """
    Decorator to add interpolation and projection steps for embedded
    DG advection.
    """
    def get_apply(self, x_in, x_out):

        if self.discretisation_option in ["embedded_dg", "recovered"]:

            def new_apply(self, x_in, x_out):

                self.pre_apply(x_in, self.discretisation_option)
                original_apply(self, self.xdg_in, self.xdg_out)
                self.post_apply(x_out, self.discretisation_option)

            return new_apply(self, x_in, x_out)

        else:

            return original_apply(self, x_in, x_out)

    return get_apply


class Advection(object, metaclass=ABCMeta):
    """
    Base class for advection schemes.

    :arg state: :class:`.State` object.
    :arg field: field to be advected
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg solver_parameters: solver_parameters
    :arg limiter: :class:`.Limiter` object.
    :arg options: :class:`.AdvectionOptions` object
    """

    def __init__(self, state, field, equation, *,
                 solver_parameters=None,
                 limiter=None, options=None):

        self.state = state
        self.field = field
        self.equation = equation().label_map(lambda t: not any(t.has_label(time_derivative, advection)), map_if_true=drop)

        if len(equation.function_space) > 1:
            idx = self.field.function_space().index
            self.equation = self.equation.label_map(lambda t: t.labels["subject"].function_space().index == idx, extract(idx), drop)

        self.ubar = Function(state.spaces("HDiv"))
        self.dt = state.timestepping.dt

        self.solver_parameters = (
            solver_parameters or {'ksp_type': 'cg',
                                  'pc_type': 'bjacobi',
                                  'sub_pc_type': 'ilu'}
        )

        self.limiter = limiter

        if options is not None:
            self.discretisation_option = options.name
            self._setup(state, self.field, options)
        else:
            self.discretisation_option = None
            self.fs = self.field.function_space()

        if self.discretisation_option is not None:
            self.equation = self.equation.label_map(all_terms,
                                                    replace_test(self.test))

        # setup required functions
        self.trial = TrialFunction(self.fs)
        self.dq = Function(self.fs)
        self.q1 = Function(self.fs)

    def _setup(self, state, field, options):

        if options.name in ["embedded_dg", "recovered"]:
            if options.embedding_space is None:
                V_elt = BrokenElement(field.function_space().ufl_element())
                self.fs = FunctionSpace(state.mesh, V_elt)
            else:
                self.fs = options.embedding_space
            self.xdg_in = Function(self.fs)
            self.xdg_out = Function(self.fs)
            self.test = TestFunction(self.fs)
            self.x_projected = Function(field.function_space())
            parameters = {'ksp_type': 'cg',
                          'pc_type': 'bjacobi',
                          'sub_pc_type': 'ilu'}
            self.Projector = Projector(self.xdg_out, self.x_projected,
                                       solver_parameters=parameters)

        if options.name == "recovered":
            # set up the necessary functions
            self.x_in = Function(field.function_space())
            x_rec = Function(options.recovered_space)
            x_brok = Function(options.broken_space)

            # set up interpolators and projectors
            self.x_rec_projector = Recoverer(self.x_in, x_rec, VDG=self.fs, boundary_method=options.boundary_method)  # recovered function
            self.x_brok_projector = Projector(x_rec, x_brok)  # function projected back
            self.xdg_interpolator = Interpolator(self.x_in + x_rec - x_brok, self.xdg_in)
            if self.limiter is not None:
                self.x_brok_interpolator = Interpolator(self.xdg_out, x_brok)
                self.x_out_projector = Recoverer(x_brok, self.x_projected)
                # when the "average" method comes into firedrake master, this will be
                # self.x_out_projector = Projector(x_brok, self.x_projected, method="average")
        elif options.name == "supg":
            self.fs = field.function_space()
            self.solver_parameters['ksp_type'] = 'gmres'

            dim = state.mesh.topological_dimension()
            if options.tau is not None:
                tau = options.tau
                assert tau.ufl_shape == (dim, dim)
            else:
                vals = [options.default*self.dt]*dim
                for component, value in options.tau_components:
                    vals[state.components.component] = value
                tau = Constant(tuple([
                    tuple(
                        [vals[j] if i == j else 0. for i, v in enumerate(vals)]
                    ) for j in range(dim)])
                )
            test = TestFunction(self.fs)
            self.test = test + dot(dot(self.ubar, tau), grad(test))

    def pre_apply(self, x_in, discretisation_option):
        """
        Extra steps to advection if using an embedded method,
        which might be either the plain embedded method or the
        recovered space advection scheme.

        :arg x_in: the input set of prognostic fields.
        :arg discretisation option: string specifying which scheme to use.
        """
        if discretisation_option == "embedded_dg":
            try:
                self.xdg_in.interpolate(x_in)
            except NotImplementedError:
                self.xdg_in.project(x_in)

        elif discretisation_option == "recovered":
            self.x_in.assign(x_in)
            self.x_rec_projector.project()
            self.x_brok_projector.project()
            self.xdg_interpolator.interpolate()

    def post_apply(self, x_out, discretisation_option):
        """
        The projection steps, returning a field to its original space
        for an embedded DG advection scheme. For the case of the
        recovered scheme, there are two options dependent on whether
        the scheme is limited or not.

        :arg x_out: the outgoing field.
        :arg discretisation_option: string specifying which option to use.
        """
        if discretisation_option == "embedded_dg":
            self.Projector.project()

        elif discretisation_option == "recovered":
            if self.limiter is not None:
                self.x_brok_interpolator.interpolate()
                self.x_out_projector.project()
            else:
                self.Projector.project()
        x_out.assign(self.x_projected)

    @abstractproperty
    def lhs(self):

        return self.equation.label_map(lambda t: t.has_label(time_derivative),
                                       map_if_true=replace_labelled("subject", self.trial, single=True),
                                       map_if_false=drop).form

    @abstractproperty
    def rhs(self):
        r = self.equation.label_map(all_terms, replace_labelled("subject", self.q1, single=True))
        r = r.label_map(lambda t: t.has_label(advection), replace_labelled("uadv", self.ubar, single=True))
        r = r.label_map(lambda t: not t.has_label(time_derivative), lambda t: self.dt*t)
        return r.form

    def update_ubar(self, uadv):
        self.ubar.assign(uadv)

    @cached_property
    def solver(self):
        # setup solver using lhs and rhs defined in derived class
        problem = LinearVariationalProblem(self.lhs, self.rhs, self.dq)
        solver_name = self.field.name()+self.equation.__class__.__name__+self.__class__.__name__
        return LinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @abstractmethod
    def apply(self, x_in, x_out):
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass


class ExplicitAdvection(Advection):
    """
    Base class for explicit advection schemes.

    :arg state: :class:`.State` object.
    :arg field: field to be advected
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg subcycles: (optional) integer specifying number of subcycles to perform
    :arg solver_parameters: solver_parameters
    :arg limiter: :class:`.Limiter` object.
    """

    def __init__(self, state, field, equation=None, *, subcycles=None,
                 solver_parameters=None, limiter=None, options=None):
        super().__init__(state, field, equation,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)

        # if user has specified a number of subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
        if subcycles is not None:
            self.dt = self.dt/subcycles
            self.ncycles = subcycles
        else:
            self.dt = self.dt
            self.ncycles = 1
        self.x = [Function(self.fs)]*(self.ncycles+1)

    @abstractmethod
    def apply_cycle(self, x_in, x_out):
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass

    @embedded_dg
    def apply(self, x_in, x_out):
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        self.x[0].assign(x_in)
        for i in range(self.ncycles):
            self.apply_cycle(self.x[i], self.x[i+1])
            self.x[i].assign(self.x[i+1])
        x_out.assign(self.x[self.ncycles-1])


class ForwardEuler(ExplicitAdvection):
    """
    Class to implement the forward Euler timestepping scheme:
    y_(n+1) = y_n + dt*L(y_n)
    where L is the advection operator
    """

    @cached_property
    def lhs(self):
        return super(ForwardEuler, self).lhs

    @cached_property
    def rhs(self):
        return super(ForwardEuler, self).rhs

    def apply_cycle(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class SSPRK3(ExplicitAdvection):
    """
    Class to implement the Strongly Structure Preserving Runge Kutta 3-stage
    timestepping method:
    y^1 = y_n + L(y_n)
    y^2 = (3/4)y_n + (1/4)(y^1 + L(y^1))
    y_(n+1) = (1/3)y_n + (2/3)(y^2 + L(y^2))
    where subscripts indicate the timelevel, superscripts indicate the stage
    number and L is the advection operator.
    """

    @cached_property
    def lhs(self):
        return super(SSPRK3, self).lhs

    @cached_property
    def rhs(self):
        return super(SSPRK3, self).rhs

    def solve_stage(self, x_in, stage):

        if stage == 0:
            self.solver.solve()
            self.q1.assign(self.dq)

        elif stage == 1:
            self.solver.solve()
            self.q1.assign(0.75*x_in + 0.25*self.dq)

        elif stage == 2:
            self.solver.solve()
            self.q1.assign((1./3.)*x_in + (2./3.)*self.dq)

        if self.limiter is not None:
            self.limiter.apply(self.q1)

    def apply_cycle(self, x_in, x_out):

        if self.limiter is not None:
            self.limiter.apply(x_in)

        self.q1.assign(x_in)
        for i in range(3):
            self.solve_stage(x_in, i)
        x_out.assign(self.q1)


class ThetaMethod(Advection):
    """
    Class to implement the theta timestepping method:
    y_(n+1) = y_n + dt*(theta*L(y_n) + (1-theta)*L(y_(n+1))) where L is the advection operator.
    """
    def __init__(self, state, field, equation, theta=0.5, solver_parameters=None):

        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super(ThetaMethod, self).__init__(state, field, equation,
                                          solver_parameters=solver_parameters)

        self.theta = theta

    @cached_property
    def lhs(self):

        l = self.equation.label_map(all_terms, replace_labelled("subject", self.trial, single=True))
        l = l.label_map(lambda t: t.has_label(advection), replace_labelled("uadv", self.ubar, single=True))
        l = l.label_map(lambda t: not t.has_label(time_derivative), lambda t: -self.theta*self.dt*t)
        return l.form

    @cached_property
    def rhs(self):

        r = self.equation.label_map(all_terms, replace_labelled("subject", self.q1, single=True))
        r = r.label_map(lambda t: t.has_label(advection), replace_labelled("uadv", self.ubar, single=True))
        r = r.label_map(lambda t: not t.has_label(time_derivative), lambda t: (1-self.theta)*self.dt*t)
        return r.form

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)
