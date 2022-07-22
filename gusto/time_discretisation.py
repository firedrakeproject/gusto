from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import (Function, NonlinearVariationalProblem, split,
                       NonlinearVariationalSolver, Projector, Interpolator,
                       BrokenElement, VectorElement, FunctionSpace,
                       TestFunction, Constant, dot, grad, as_ufl, MixedElement,
                       DirichletBC)
from firedrake.formmanipulation import split_form
from firedrake.utils import cached_property
import ufl
from gusto.configuration import logger, DEBUG, TransportEquationType
from gusto.labels import (time_derivative, transporting_velocity, prognostic, subject,
                          transport, ibp_label, replace_subject, replace_test_function)
from gusto.recovery import Recoverer
from gusto.fml.form_manipulation_labelling import Term, all_terms, drop
from gusto.transport_forms import advection_form, continuity_form


__all__ = ["ForwardEuler", "BackwardEuler", "SSPRK3", "RK4", "Heun", "ThetaMethod", "ImplicitMidpoint"]


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
    Decorator to add interpolation and projection steps for embedded DG method.
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


class TimeDiscretisation(object, metaclass=ABCMeta):
    """
    Base class for time discretisation schemes.

    :arg state: :class:`.State` object.
    :arg field: field to be evolved
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg solver_parameters: solver_parameters
    :arg limiter: :class:`.Limiter` object.
    :arg options: :class:`.DiscretisationOptions` object
    """

    def __init__(self, state, field_name=None, solver_parameters=None,
                 limiter=None, options=None):

        self.state = state
        self.field_name = field_name

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

    def setup(self, equation, uadv=None, apply_bcs=True, *active_labels):

        self.residual = equation.residual

        if self.field_name is not None:
            self.idx = equation.field_names.index(self.field_name)
            self.fs = self.state.fields(self.field_name).function_space()
            self.residual = self.residual.label_map(
                lambda t: t.get(prognostic) == self.field_name,
                lambda t: Term(
                    split_form(t.form)[self.idx].form,
                    t.labels),
                drop)
            bcs = equation.bcs[self.field_name]

        else:
            self.field_name = equation.field_name
            self.fs = equation.function_space
            self.idx = None
            if type(self.fs.ufl_element()) is MixedElement:
                bcs = [bc for _, bcs in equation.bcs.items() for bc in bcs]
            else:
                bcs = equation.bcs[self.field_name]

        if len(active_labels) > 0:
            self.residual = self.residual.label_map(
                lambda t: any(t.has_label(time_derivative, *active_labels)),
                map_if_false=drop)

        options = self.options

        # -------------------------------------------------------------------- #
        # Routines relating to transport
        # -------------------------------------------------------------------- #

        if hasattr(self.options, 'ibp'):
            self.replace_transport_term()
        self.replace_transporting_velocity(uadv)

        # -------------------------------------------------------------------- #
        # Wrappers for embedded / recovery methods
        # -------------------------------------------------------------------- #

        if self.discretisation_option in ["embedded_dg", "recovered"]:
            # construct the embedding space if not specified
            if options.embedding_space is None:
                V_elt = BrokenElement(self.fs.ufl_element())
                self.fs = FunctionSpace(self.state.mesh, V_elt)
            else:
                self.fs = options.embedding_space
            self.xdg_in = Function(self.fs)
            self.xdg_out = Function(self.fs)
            if self.idx is None:
                self.x_projected = Function(equation.function_space)
            else:
                self.x_projected = Function(self.state.fields(self.field_name).function_space())
            new_test = TestFunction(self.fs)
            parameters = {'ksp_type': 'cg',
                          'pc_type': 'bjacobi',
                          'sub_pc_type': 'ilu'}

        # -------------------------------------------------------------------- #
        # Make boundary conditions
        # -------------------------------------------------------------------- #

        if not apply_bcs:
            self.bcs = None
        elif self.discretisation_option in ["embedded_dg", "recovered"]:
            # Transfer boundary conditions onto test function space
            self.bcs = [DirichletBC(self.fs, bc.function_arg, bc.sub_domain) for bc in bcs]
        else:
            self.bcs = bcs

        # -------------------------------------------------------------------- #
        # Modify test function for SUPG methods
        # -------------------------------------------------------------------- #

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
                self.solver_parameters = {'ksp_type': 'gmres',
                                          'pc_type': 'bjacobi',
                                          'sub_pc_type': 'ilu'}

            test = TestFunction(self.fs)
            new_test = test + dot(dot(uadv, tau), grad(test))

        if self.discretisation_option is not None:
            # replace the original test function with one defined on
            # the embedding space, as this is the space where the
            # the problem will be solved
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
            self.x_in = Function(self.state.fields(self.field_name).function_space())
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
        self.dq = Function(self.fs)
        self.q1 = Function(self.fs)

    def pre_apply(self, x_in, discretisation_option):
        """
        Extra steps to discretisation if using an embedded method,
        which might be either the plain embedded method or the
        recovered space scheme.

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
        for an embedded DG scheme. For the case of the
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
            map_if_true=replace_subject(self.dq, self.idx),
            map_if_false=drop)

        return l.form

    @abstractproperty
    def rhs(self):
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.q1, self.idx))

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: -self.dt*t)

        return r.form

    def replace_transport_term(self):
        """
        This routine allows the default transport term to be replaced with a
        different one, specified through the transport options.

        This is necessary because when the prognostic equations are declared,
        the whole transport
        """

        # Extract transport term of equation
        old_transport_term_list = self.residual.label_map(
            lambda t: t.has_label(transport), map_if_false=drop)

        # If there are more transport terms, extract only the one for this variable
        if len(old_transport_term_list.terms) > 1:
            raise NotImplementedError('Cannot replace transport terms when there are more than one')

        # Then we should only have one transport term
        old_transport_term = old_transport_term_list.terms[0]

        # If the transport term has an ibp label, then it could be replaced
        if old_transport_term.has_label(ibp_label) and hasattr(self.options, 'ibp'):
            # Do the options specify a different ibp to the old transport term?
            if old_transport_term.labels['ibp'] != self.options.ibp:
                # Set up a new transport term
                field = self.state.fields(self.field_name)
                test = TestFunction(self.fs)

                # Set up new transport term (depending on the type of transport equation)
                if old_transport_term.labels['transport'] == TransportEquationType.advective:
                    new_transport_term = advection_form(self.state, test, field, ibp=self.options.ibp)
                elif old_transport_term.labels['transport'] == TransportEquationType.conservative:
                    new_transport_term = continuity_form(self.state, test, field, ibp=self.options.ibp)
                else:
                    raise NotImplementedError(f'Replacement of transport term not implemented yet for {old_transport_term.labels["transport"]}')

                # Finally, drop the old transport term and add the new one
                self.residual = self.residual.label_map(
                    lambda t: t.has_label(transport), map_if_true=drop)
                self.residual += subject(new_transport_term, field)

    def replace_transporting_velocity(self, uadv):
        # replace the transporting velocity in any terms that contain it
        if any([t.has_label(transporting_velocity) for t in self.residual]):
            assert uadv is not None
            if uadv == "prognostic":
                self.residual = self.residual.label_map(
                    lambda t: t.has_label(transporting_velocity),
                    map_if_true=lambda t: Term(ufl.replace(
                        t.form, {t.get(transporting_velocity): split(t.get(subject))[0]}), t.labels)
                )
            else:
                self.residual = self.residual.label_map(
                    lambda t: t.has_label(transporting_velocity),
                    map_if_true=lambda t: Term(ufl.replace(
                        t.form, {t.get(transporting_velocity): uadv}), t.labels)
                )
            self.residual = transporting_velocity.update_value(self.residual, uadv)

    @cached_property
    def solver(self):
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.dq, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
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


class ExplicitTimeDiscretisation(TimeDiscretisation):
    """
    Base class for explicit time discretisations.

    :arg state: :class:`.State` object.
    :arg field: field to be evolved
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg subcycles: (optional) integer specifying number of subcycles to perform
    :arg solver_parameters: solver_parameters
    :arg limiter: :class:`.Limiter` object.
    """

    def __init__(self, state, field_name=None, subcycles=None,
                 solver_parameters=None, limiter=None, options=None):
        super().__init__(state, field_name,
                         solver_parameters=solver_parameters,
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


class ForwardEuler(ExplicitTimeDiscretisation):
    """
    Class to implement the forward Euler timestepping scheme:
    y_(n+1) = y_n + dt*L(y_n)
    where L is the operator
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


class SSPRK3(ExplicitTimeDiscretisation):
    """
    Class to implement the Strongly Structure Preserving Runge Kutta 3-stage
    timestepping method:
    y^1 = y_n + L(y_n)
    y^2 = (3/4)y_n + (1/4)(y^1 + L(y^1))
    y_(n+1) = (1/3)y_n + (2/3)(y^2 + L(y^2))
    where subscripts indicate the time-level, superscripts indicate the stage
    number and L is the operator.
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


class RK4(ExplicitTimeDiscretisation):
    """
    Class to implement the 4-stage Runge-Kutta timestepping method:
    k1 = f(y_n)
    k2 = f(y_n + 1/2*dt*k1)
    k3 = f(y_n + 1/2*dt*k2)
    k4 = f(y_n + dt*k3)
    y_(n+1) = y_n + (1/6) * dt * (k1 + 2*k2 + 2*k3 + k4)
    where subscripts indicate the timelevel, superscripts indicate the stage
    number and f is the RHS.
    """

    def setup(self, equation, uadv, *active_labels):

        super().setup(equation, uadv, *active_labels)

        self.k1 = Function(self.fs)
        self.k2 = Function(self.fs)
        self.k3 = Function(self.fs)
        self.k4 = Function(self.fs)

    @cached_property
    def lhs(self):
        l = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.dq, self.idx),
            map_if_false=drop)

        return l.form

    @cached_property
    def rhs(self):
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.q1, self.idx))

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=drop,
            map_if_false=lambda t: -1*t)

        return r.form

    def solve_stage(self, x_in, stage):

        if stage == 0:
            self.solver.solve()
            self.k1.assign(self.dq)
            self.q1.assign(x_in + 0.5 * self.dt * self.k1)

        elif stage == 1:
            self.solver.solve()
            self.k2.assign(self.dq)
            self.q1.assign(x_in + 0.5 * self.dt * self.k2)

        elif stage == 2:
            self.solver.solve()
            self.k3.assign(self.dq)
            self.q1.assign(x_in + self.dt * self.k3)

        elif stage == 3:
            self.solver.solve()
            self.k4.assign(self.dq)
            self.q1.assign(x_in + 1/6 * self.dt * (self.k1 + 2*self.k2 + 2*self.k3 + self.k4))

    def apply_cycle(self, x_in, x_out):
        if self.limiter is not None:
            self.limiter.apply(x_in)

        self.q1.assign(x_in)

        for i in range(4):
            self.solve_stage(x_in, i)
        x_out.assign(self.q1)


class Heun(ExplicitTimeDiscretisation):
    """
    Class to implement Heun's timestepping method:
    y^1 = L(y_n)
    y_(n+1) = (1/2)y_n + (1/2)Ly^1)
    where subscripts indicate the timelevel, superscripts indicate the stage
    number and L is the advection operator.
    """
    @cached_property
    def lhs(self):
        return super(Heun, self).lhs

    @cached_property
    def rhs(self):
        return super(Heun, self).rhs

    def solve_stage(self, x_in, stage):
        if stage == 0:
            self.solver.solve()
            self.q1.assign(self.dq)

        elif stage == 1:
            self.solver.solve()
            self.q1.assign(0.5 * x_in + 0.5 * (self.dq))

    def apply_cycle(self, x_in, x_out):

        if self.limiter is not None:
            self.limiter.apply(x_in)

        self.q1.assign(x_in)
        for i in range(2):
            self.solve_stage(x_in, i)
        x_out.assign(self.q1)


class BackwardEuler(TimeDiscretisation):

    @property
    def lhs(self):
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.dq, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.dt*t)

        return l.form

    @property
    def rhs(self):

        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.q1, self.idx),
            map_if_false=drop)

        return r.form

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class ThetaMethod(TimeDiscretisation):
    """
    Class to implement the theta timestepping method:
    y_(n+1) = y_n + dt*(theta*L(y_n) + (1-theta)*L(y_(n+1))) where L is the operator.
    """
    def __init__(self, state, field_name=None, theta=None,
                 solver_parameters=None, options=None):

        if theta is None:
            raise ValueError("please provide a value for theta between 0 and 1")
        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super().__init__(state, field_name,
                         solver_parameters=solver_parameters,
                         options=options)

        self.theta = theta

    @cached_property
    def lhs(self):
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.dq, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.theta*self.dt*t)

        return l.form

    @cached_property
    def rhs(self):
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.q1, self.idx))
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
    where L is the operator.
    """
    def __init__(self, state, field_name=None, solver_parameters=None,
                 options=None):
        super().__init__(state, field_name, theta=0.5,
                         solver_parameters=solver_parameters,
                         options=options)
