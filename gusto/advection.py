from abc import ABCMeta, abstractmethod
from firedrake import (Function, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, Projector, Interpolator,
                       TestFunction, TrialFunction, FunctionSpace,
                       TrialFunctions, action, as_ufl,
                       BrokenElement, Constant, dot, grad)
from firedrake.utils import cached_property
import ufl
from gusto.configuration import logger, DEBUG
from gusto.form_manipulation_labelling import (all_terms, has_labels, index,
                                               advecting_velocity, subject,
                                               time_derivative, drop,
                                               replace_test, replace_labelled,
                                               extract, Term,
                                               explicit, implicit)
from gusto.recovery import Recoverer
from gusto.transport_equation import is_cg


__all__ = ["BackwardEuler", "ForwardEuler", "SSPRK3", "ThetaMethod", "ImplicitMidpoint"]


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

    :arg solver_parameters: (optional) solver_parameters
    :arg limiter: (optional) :class:`.Limiter` object.
    :arg options: (optional) :class:`.AdvectionOptions` object
    """

    def __init__(self, *,
                 solver_parameters=None,
                 limiter=None, options=None):

        self._initialised = False

        self.solver_parameters = solver_parameters

        self.limiter = limiter

        self.options = options

    def _label_terms(self):
        """
        Function to label the terms of the equation form. This function is
        specific to the timestepping scheme so is implemented in the child
        classes.
        """
        pass

    def _setup_from_options(self, state, fs, options):
        """
        This function deals with any spatial discretisation options
        specified in the :class:`.AdvectionOptions` object passed in
        when the class was instantiated. This includes setting up the
        functions and projections required for embedded DG and recovered
        advection schemes; constructing the supg test function; and
        replacing the test function in the equation form with that defined
        on the correct space (for embedded DG and recovered schemes) or
        with an amended test function (for supg).

        :arg state: a :class:`.State` object
        :arg fs: the function space of the field
        :arg options: an :class:`AdvectionOptions` object, containing
        options related to the spatial discretisation
        """
        self.discretisation_option = options.name

        if options.name in ["embedded_dg", "recovered"]:
            # construct the embedding space if not specified
            if options.embedding_space is None:
                V_elt = BrokenElement(fs.ufl_element())
                self.fs = FunctionSpace(state.mesh, V_elt)
            else:
                self.fs = options.embedding_space
            # make functions and projector for moving between the
            # embedding and the original spaces
            self.xdg_in = Function(self.fs)
            self.xdg_out = Function(self.fs)
            self.x_projected = Function(fs)
            parameters = {'ksp_type': 'cg',
                          'pc_type': 'bjacobi',
                          'sub_pc_type': 'ilu'}
            self.Projector = Projector(self.xdg_out, self.x_projected,
                                       solver_parameters=parameters)
            # replace the original test function with one defined on
            # the embedding space, as this is the space where the
            # advection occurs
            test = TestFunction(self.fs)
            self.equation = self.equation.label_map(all_terms,
                                                    replace_test(test))

        if options.name == "recovered":
            # set up the necessary functions
            self.x_in = Function(fs)
            x_rec = Function(options.recovered_space)
            x_brok = Function(options.broken_space)

            # set up interpolators and projectors
            self.x_rec_projector = Recoverer(self.x_in, x_rec, VDG=fs, boundary_method=options.boundary_method)  # recovered function
            self.x_brok_projector = Projector(x_rec, x_brok)  # function projected back
            self.xdg_interpolator = Interpolator(self.x_in + x_rec - x_brok, self.xdg_in)
            if self.limiter is not None:
                self.x_brok_interpolator = Interpolator(self.xdg_out, x_brok)
                self.x_out_projector = Recoverer(x_brok, self.x_projected)
                # when the "average" method comes into firedrake master, this will be
                # self.x_out_projector = Projector(x_brok, self.x_projected, method="average")

        elif options.name == "supg":

            self.fs = fs
            # construct tau, if it is not specified
            dim = state.mesh.topological_dimension()
            if options.tau is not None:
                # if tau is provided, check that is has the right size
                tau = options.tau
                assert as_ufl(tau).ufl_shape == (dim, dim), "Provided tau has incorrect shape!"
            else:
                # create tuple of default values of size dim
                default_vals = [options.default*self.dt]*dim
                # check for directions is which the space is discontinuous
                # so that we don't apply supg in that direction
                if is_cg(fs):
                    vals = default_vals
                else:
                    space = fs.ufl_element().sobolev_space()
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

            # replace test function with supg test function
            test = TestFunction(fs)
            default_uadv = Function(state.spaces("HDiv"))

            def replace_with_supg_test(t):
                uadv = t.get(advecting_velocity) or default_uadv
                test = t.form.arguments()[0]
                new_test = test + dot(dot(uadv, tau), grad(test))
                return advecting_velocity(Term(ufl.replace(t.form, {test: new_test}), t.labels), uadv)

            self.equation = self.equation.label_map(
                all_terms, replace_with_supg_test)

            # update default ksp_type
            self.default_ksp_type = 'gmres'

    def setup(self, state, equation, dt, *active_labels,
              field_name=None, u_advecting=None):
        """
        This function is called from the Timstepper class. At this point
        we have all the information we need to extract the required parts
        of the form, replace the test function (if required by the spatial
        discretisation options) and setup the functions and solver parameters
        for the timestepping scheme.

        :arg state: a :class:`.State` object
        :arg equation: a :class:`.PrognosticEquation` object
        :arg dt: the timestep
        :arg active_labels: :class:`Label` object(s) specifying the label(s)
             of terms in the form that this scheme should be applied to
        :arg field_name: (optional) str, naming the prognostic field. This
             is necessary when solving for a subfield of a mixed system.
        :arg u_advecting: (optional) a ufl expression for the advecting
             velocity. If not present then treat any parts of the form
             labelled advecting_velocity as the equation subject (i.e.
             they are part of the solution to be computed by this scheme).
        """
        if self._initialised:
            raise RuntimeError("Trying to setup an advection scheme that has already been setup.")

        self.dt = dt
        if field_name is None:
            self.field_name = equation.field_name
            fs = equation.function_space
        else:
            self.field_name = field_name
            fs = state.fields(field_name).function_space()

        # store just the form
        self.equation = equation()

        # is the equation is defined on a mixed function space
        mixed_equation = len(equation.function_space) > 1

        # is the prognostic field is defined on a mixed
        # function space
        mixed_function = len(fs) > 1

        # if the equation in defined on a mixed function space, but
        # the prognostic field isn't, then extract the parts of the
        # equation form that involve this prognostic field.
        if mixed_equation and not mixed_function:
            idx = fs.index
            self.equation = self.equation.label_map(
                lambda t: t.get(index) == idx, extract(idx), drop)

        # select labelled terms from the equation if active_labels are
        # specified
        if len(active_labels) > 0:
            self.equation = self.equation.label_map(
                has_labels(time_derivative, *active_labels),
                map_if_false=drop)

        # if options have been specified via an AdvectionOptions
        # class, now is the time to apply them
        if self.options is not None:
            if mixed_equation and mixed_function:
                raise NotImplementedError("%s options not implemented for mixed problems" % self.options.name)
            else:
                self._setup_from_options(state, fs, self.options)
        else:
            self.discretisation_option = None
            self.fs = fs

        # replace the advecting velocity in any terms that contain it
        if any([t.has_label(advecting_velocity) for t in self.equation]):
            # setup advecting velocity
            if u_advecting is None:
                # the advecting velocity is calculated as part of this
                # timestepping scheme and must be replaced with the
                # correct part of the term's subject
                assert mixed_function, "We expect the advecting velocity to be specified unless we are applying this timestepping scheme to a mixed system of equations"
                self.equation = self.equation.label_map(
                    has_labels(advecting_velocity),
                    replace_labelled(subject, advecting_velocity))
            else:
                # the advecting velocity is fixed over the timestep
                # and is specified by the advecting_velocity property
                # of the timestepping class
                self.equation = self.equation.label_map(
                    has_labels(advecting_velocity),
                    replace_labelled(u_advecting, advecting_velocity))

        # setup required functions
        self.q1 = Function(self.fs)
        self.result = Function(self.fs)

        # setup trial function(s) and solver parameters.
        if mixed_equation and mixed_function:
            self.trial = TrialFunctions(self.fs)
            # for a mixed system the default solver parameters are an
            # attribute of the :class:`PrognosticMixedEquations` object
            default_solver_params = equation.solver_parameters
        else:
            self.trial = TrialFunction(self.fs)
            # the default ksp_type will be different if the matrix is
            # not symmetric, so if we are using supg or a theta method
            try:
                ksp_type = self.default_ksp_type
            except AttributeError:
                ksp_type = 'cg'
            default_solver_params = {'ksp_type': ksp_type,
                                     'pc_type': 'bjacobi',
                                     'sub_pc_type': 'ilu'}

        # use default solver parameters if the user has not specified
        # any on instantiating this class
        if self.solver_parameters is None:
            self.solver_parameters = default_solver_params
        if logger.isEnabledFor(DEBUG):
            self.solver_parameters["ksp_monitor_true_residual"] = True

        # label the terms explicit or implicit
        self._label_terms()

        self._initialised = True

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

    @property
    def lhs(self):
        l = self.equation.label_map(
            has_labels(time_derivative, implicit),
            map_if_true=replace_labelled(self.trial, subject),
            map_if_false=drop)
        l = l.label_map(has_labels(time_derivative),
                        map_if_false=lambda t: self.dt*t)
        return l.form

    @property
    def rhs(self):

        r = self.equation.label_map(
            has_labels(time_derivative, explicit),
            map_if_true=replace_labelled(self.q1, subject),
            map_if_false=drop)

        r = r.label_map(has_labels(time_derivative),
                        map_if_false=lambda t: -self.dt*t)
        return r.form

    @cached_property
    def solver(self):
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(action(self.lhs, self.result)-self.rhs, self.result)
        solver_name = self.field_name+self.equation.__class__.__name__+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @abstractmethod
    def apply(self, x_in, x_out):
        """
        Applies advection scheme
        """
        pass


class BackwardEuler(Advection):
    """
    Class to implement the backward Euler timestepping scheme:
    y_(n+1) - dt*F(y_(n+1)) = y_n
    """

    def _label_terms(self):
        """
        Labels all terms implicit
        """
        self.equation = self.equation.label_map(
            has_labels(time_derivative),
            map_if_false=lambda t: implicit(t))

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.result)


class ExplicitAdvection(Advection):
    """
    Base class for explicit advection schemes.

    :arg subcycles: (optional) integer specifying number of subcycles to perform
    :arg solver_parameters: (optional) solver_parameters
    :arg limiter: (optional) :class:`.Limiter` object.
    :arg options: (optional) :class:`.AdvectionOptions` object
    """

    def __init__(self, subcycles=None, solver_parameters=None,
                 limiter=None, options=None):

        self.subcycles = subcycles
        super().__init__(solver_parameters=solver_parameters,
                         limiter=limiter, options=options)

    def setup(self, state, equation, dt, *active_labels,
              field_name=None, u_advecting=None):

        # if user has specified a number of subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
        if self.subcycles is not None:
            dt = dt/self.subcycles
            self.ncycles = self.subcycles
        else:
            dt = dt
            self.ncycles = 1

        super().setup(state, equation, dt, *active_labels,
                      field_name=field_name, u_advecting=u_advecting)

        # setup functions to store the result of each cycle
        self.x = [Function(self.fs)]*(self.ncycles+1)

    def _label_terms(self):
        """
        Labels all terms explicit
        """
        self.equation = self.equation.label_map(
            has_labels(time_derivative),
            map_if_false=lambda t: explicit(t))

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
        Function takes x as input, computes F(x) as defined by the equation,
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
    y_(n+1) = y_n + dt*F(y_n)
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
        x_out.assign(self.result)


class SSPRK3(ExplicitAdvection):
    """
    Class to implement the Strongly Structure Preserving Runge Kutta 3-stage
    timestepping method:
    y^1 = y_n + F(y_n)
    y^2 = (3/4)y_n + (1/4)(y^1 + F(y^1))
    y_(n+1) = (1/3)y_n + (2/3)(y^2 + F(y^2))
    where subscripts indicate the timelevel, superscripts indicate the stage
    number.
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
            self.q1.assign(self.result)

        elif stage == 1:
            self.solver.solve()
            self.q1.assign(0.75*x_in + 0.25*self.result)

        elif stage == 2:
            self.solver.solve()
            self.q1.assign((1./3.)*x_in + (2./3.)*self.result)

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
    y_(n+1) + dt*theta*F(y_(n+1))) = y_n - dt*(1-theta)*F(y_n)
    to solve the equation dy/dt + F(y) = 0.

    :arg theta: value of theta
    :arg solver_parameters: (optional) solver_parameters
    :arg options: (optional) :class:`.AdvectionOptions` object
    """
    def __init__(self, theta, solver_parameters=None, options=None):

        super().__init__(solver_parameters=solver_parameters, options=options)

        self.theta = theta

        # theta method leads to asymmetric matrix, per lhs
        # function below, so don't use CG
        self.default_ksp_type = 'gmres'

    @cached_property
    def lhs(self):

        l = self.equation.label_map(all_terms,
                                    replace_labelled(self.trial, subject))
        l = l.label_map(has_labels(time_derivative),
                        map_if_false=lambda t: self.theta*self.dt*t)
        return l.form

    @cached_property
    def rhs(self):

        r = self.equation.label_map(all_terms,
                                    replace_labelled(self.q1, subject))
        r = r.label_map(has_labels(time_derivative),
                        map_if_false=lambda t: -(1-self.theta)*self.dt*t)
        return r.form

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.result)


class ImplicitMidpoint(ThetaMethod):
    """
    Class to implement the implicit midpoint timestepping method:
    y_(n+1) + 0.5*dt*F(y_(n+1))) = y_n - 0.5*dt*F(y_n)
    to solve the equation dy/dt + F(y) = 0.

    :arg solver_parameters: (optional) solver_parameters
    :arg options: (optional) :class:`.AdvectionOptions` object
    """
    def __init__(self, solver_parameters=None, options=None):

        super().__init__(theta=0.5, solver_parameters=solver_parameters,
                         options=options)

    def apply(self, x_in, x_out):
        super().apply(x_in, x_out)
