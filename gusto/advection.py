from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import (Function, TrialFunction, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, Projector, Interpolator,
                       BrokenElement, VectorElement, FunctionSpace,
                       TestFunction, action, Constant, dot, grad, as_ufl)
from firedrake.utils import cached_property
import ufl
from gusto.configuration import logger, DEBUG
from gusto.form_manipulation_labelling import (Term, drop, time_derivative,
                                               advecting_velocity,
                                               all_terms, replace_subject,
                                               replace_test_function)
from gusto.recovery import Recoverer


__all__ = ["ForwardEuler", "BackwardEuler", "SSPRK3", "ThetaMethod", "ImplicitMidpoint"]


def is_cg(V):
    """
    Function to check if a given space, V, is CG. Broken elements are
    always discontinuous; for vector elements we check the names of
    the sobolev spaces of the subelements and for all other elements
    we just check the sobolev space name.
    """
    ele = V.ufl_element()
    if isinstance(ele, BrokenElement):
        return False
    elif type(ele) == VectorElement:
        return all([e.sobolev_space().name == "H1" for e in ele._sub_elements])
    else:
        return V.ufl_element().sobolev_space().name == "H1"


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

    def __init__(self, state, solver_parameters=None, limiter=None, options=None):

        self.state = state

        self.dt = self.state.dt

        self.limiter = limiter

        self.options = options
        if options is not None:
            self.discretisation_option = options.name
        else:
            self.discretisation_option = None

        # get default solver options if none passed in
        if solver_parameters is None:
            self.solver_parameters = {'ksp_type': 'cg',
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}
        else:
            self.solver_parameters = solver_parameters
            if logger.isEnabledFor(DEBUG):
                self.solver_parameters["ksp_monitor_true_residual"] = None

    def setup(self, equation, uadv=None, *active_labels):

        self.equation = equation

        if len(active_labels) > 0:
            self.residual = equation.residual.label_map(
                lambda t: any(t.has_label(time_derivative, *active_labels)),
                map_if_false=drop)
        else:
            self.residual = equation.residual

        options = self.options

        if uadv is not None:
            self.replace_advecting_velocity(uadv)

        if self.discretisation_option in ["embedded_dg", "recovered"]:
            # construct the embedding space if not specified
            if options.embedding_space is None:
                V_elt = BrokenElement(equation.function_space.ufl_element())
                self.fs = FunctionSpace(self.state.mesh, V_elt)
            else:
                self.fs = options.embedding_space
            self.xdg_in = Function(self.fs)
            self.xdg_out = Function(self.fs)
            self.x_projected = Function(equation.function_space)
            new_test = TestFunction(self.fs)
            parameters = {'ksp_type': 'cg',
                          'pc_type': 'bjacobi',
                          'sub_pc_type': 'ilu'}
        else:
            self.fs = equation.function_space

        if self.discretisation_option == "supg":
            # construct tau, if it is not specified
            dim = self.state.mesh.topological_dimension()
            if options.tau is not None:
                # if tau is provided, check that is has the right size
                tau = options.tau
                assert as_ufl(tau).ufl_shape == (dim, dim), "Provided tau has incorrect shape!"
            else:
                # create tuple of default values of size dim
                default_vals = [options.default*self.dt]*dim
                # check for directions is which the space is discontinuous
                # so that we don't apply supg in that direction
                if is_cg(self.fs):
                    vals = default_vals
                else:
                    space = self.fs.ufl_element().sobolev_space()
                    if space.name in ["HDiv", "DirectionalH"]:
                        vals = [default_vals[i] if space[i].name == "H1"
                                else 0. for i in range(dim)]
                    else:
                        raise ValueError("I don't know what to do with space %s" % space)
                tau = Constant(tuple([
                    tuple(
                        [vals[j] if i == j else 0. for i, v in enumerate(vals)]
                    ) for j in range(dim)])
                )

            test = TestFunction(self.fs)
            new_test = test + dot(dot(uadv, tau), grad(test))

        if self.discretisation_option is not None:
            # replace the original test function with one defined on
            # the embedding space, as this is the space where the
            # advection occurs
            self.residual = self.residual.label_map(
                all_terms,
                map_if_true=replace_test_function(new_test))

        if self.discretisation_option == "embedded_dg":
            if self.limiter is None:
                self.x_out_projector = Projector(self.xdg_out, self.x_projected,
                                                 solver_parameters=parameters)
            else:
                self.x_out_projector = Recoverer(self.xdg_out, self.x_projected)

        if self.discretisation_option == "recovered":
            # set up the necessary functions
            self.x_in = Function(equation.function_space)
            x_rec = Function(options.recovered_space)
            x_brok = Function(options.broken_space)

            # set up interpolators and projectors
            self.x_rec_projector = Recoverer(self.x_in, x_rec, VDG=self.fs, boundary_method=options.boundary_method)  # recovered function
            self.x_brok_projector = Projector(x_rec, x_brok)  # function projected back
            self.xdg_interpolator = Interpolator(self.x_in + x_rec - x_brok, self.xdg_in)
            if self.limiter is not None:
                self.x_brok_interpolator = Interpolator(self.xdg_out, x_brok)
                self.x_out_projector = Recoverer(x_brok, self.x_projected)
            else:
                self.x_out_projector = Projector(self.xdg_out, self.x_projected)

        # setup required functions
        self.trial = TrialFunction(self.fs)
        self.dq = Function(self.fs)
        self.q1 = Function(self.fs)

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
        if discretisation_option == "recovered" and self.limiter is not None:
            self.x_brok_interpolator.interpolate()
        self.x_out_projector.project()
        x_out.assign(self.x_projected)

    @abstractproperty
    def lhs(self):
        l = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.trial),
            map_if_false=drop)

        return l.form

    @abstractproperty
    def rhs(self):
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.q1))

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: -self.dt*t)

        return r.form

    def replace_advecting_velocity(self, uadv):
        # replace the advecting velocity in any terms that contain it
        if any([t.has_label(advecting_velocity) for t in self.residual]):
            self.residual = self.residual.label_map(
                lambda t: t.has_label(advecting_velocity),
                map_if_true=lambda t: Term(ufl.replace(
                    t.form, {t.get(advecting_velocity): uadv}), t.labels)
            )
            self.residual = advecting_velocity.update_value(self.residual, uadv)

    @cached_property
    def solver(self):
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(action(self.lhs, self.dq)-self.rhs, self.dq)
        solver_name = self.equation.field_name+self.equation.__class__.__name__+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

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

    def __init__(self, state, subcycles=None,
                 solver_parameters=None, limiter=None, options=None):
        super().__init__(state, solver_parameters=solver_parameters,
                         limiter=limiter, options=options)

        self.subcycles = subcycles

    def setup(self, equation, uadv, *active_labels):

        super().setup(equation, uadv, *active_labels)

        # if user has specified a number of subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
        if self.subcycles is not None:
            self.dt = self.dt/self.subcycles
            self.ncycles = self.subcycles
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


class BackwardEuler(Advection):

    @property
    def lhs(self):
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.trial))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.dt*t)

        return l.form

    @property
    def rhs(self):

        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.q1),
            map_if_false=drop)

        return r.form

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class ThetaMethod(Advection):
    """
    Class to implement the theta timestepping method:
    y_(n+1) = y_n + dt*(theta*L(y_n) + (1-theta)*L(y_(n+1))) where L is the advection operator.
    """
    def __init__(self, state, theta=None, solver_parameters=None, options=None):

        if theta is None:
            raise ValueError("please provide a value for theta between 0 and 1")
        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super().__init__(state, solver_parameters=solver_parameters,
                         options=options)

        self.theta = theta

    @cached_property
    def lhs(self):
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.trial))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.theta*self.dt*t)

        return l.form

    @cached_property
    def rhs(self):
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.q1))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -(1-self.theta)*self.dt*t)

        return r.form

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class ImplicitMidpoint(ThetaMethod):
    """
    Class to implement the implicit midpoint timestepping method, i.e. the
    theta method with theta=0.5:
    y_(n+1) = y_n + 0.5*dt*(L(y_n) + L(y_(n+1)))
    where L is the advection operator.
    """
    def __init__(self, state, solver_parameters=None, options=None):
        super().__init__(state, theta=0.5, solver_parameters=solver_parameters,
                         options=options)
