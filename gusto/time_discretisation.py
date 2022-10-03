u"""
Objects for discretising time derivatives.

Time discretisation objects discretise ∂y/∂t = F(y), for variable y, time t and
operator F.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import (Function, NonlinearVariationalProblem, split,
                       NonlinearVariationalSolver, Projector, Interpolator,
                       BrokenElement, VectorElement, FunctionSpace,
                       TestFunction, Constant, dot, grad, as_ufl, MixedElement,
                       DirichletBC)
from firedrake.formmanipulation import split_form
from firedrake.utils import cached_property
import ufl
from gusto.configuration import (logger, DEBUG, TransportEquationType,
                                 EmbeddedDGOptions, RecoveredOptions)
from gusto.labels import (time_derivative, transporting_velocity, prognostic, subject,
                          transport, ibp_label, replace_subject, replace_test_function)
from gusto.recovery import Recoverer
from gusto.fml.form_manipulation_labelling import Term, all_terms, drop
from gusto.transport_forms import advection_form, continuity_form


__all__ = ["ForwardEuler", "BackwardEuler", "SSPRK3", "RK4", "Heun",
           "ThetaMethod", "ImplicitMidpoint", "BDF2"]


def is_cg(V):
    """
    Checks if a :class:`FunctionSpace` is continuous.

    Function to check if a given space, V, is CG. Broken elements are always
    discontinuous; for vector elements we check the names of the Sobolev spaces
    of the subelements and for all other elements we just check the Sobolev
    space name.

    Args:
        V (:class:`FunctionSpace`): the space to check.
    """
    ele = V.ufl_element()
    if isinstance(ele, BrokenElement):
        return False
    elif type(ele) == VectorElement:
        return all([e.sobolev_space().name == "H1" for e in ele._sub_elements])
    else:
        return V.ufl_element().sobolev_space().name == "H1"


def embedded_dg(original_apply):
    """Decorator to add steps for embedded DG method."""
    def get_apply(self, x_out, x_in):

        if self.discretisation_option in ["embedded_dg", "recovered"]:

            def new_apply(self, x_out, x_in):

                self.pre_apply(x_in, self.discretisation_option)
                original_apply(self, self.xdg_out, self.xdg_in)
                self.post_apply(x_out, self.discretisation_option)

            return new_apply(self, x_out, x_in)

        else:

            return original_apply(self, x_out, x_in)

    return get_apply


class TimeDiscretisation(object, metaclass=ABCMeta):
    """Base class for time discretisation schemes."""

    def __init__(self, state, field_name=None, solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            state (:class:`State`): the model's state object.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
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
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            uadv (:class:`ufl.Expr`, optional): the transporting velocity.
                Defaults to None.
            apply_bcs (bool, optional): whether to apply the equation's boundary
                conditions. Defaults to True.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
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
        self.x_out = Function(self.fs)
        self.x1 = Function(self.fs)

    def pre_apply(self, x_in, discretisation_option):
        """
        Extra steps to the discretisation if using a "wrapper" method.

        Performs extra steps before the generic apply method if the whole method
        is a "wrapper" around some existing discretisation. For instance, if
        using an embedded or recovered method this routine performs the
        transformation to the function space in which the discretisation is
        computed.

        Args:
            x_in (:class:`Function`): the original field to be evolved.
            discretisation_option (str): specifies the "wrapper" method.
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

        else:
            raise ValueError(
                f'discretisation_option {discretisation_option} not recognised')

    def post_apply(self, x_out, discretisation_option):
        """
        Extra steps to the discretisation if using a "wrapper" method.

        Performs projection steps after the generic apply method if the whole
        method is a "wrapper" around some existing discretisation. This
        generally returns a field to its original space. For the case of the
        recovered scheme, there are two options dependent on whether
        the scheme is limited or not.

        Args:
            x_out (:class:`Function`): the outgoing field to be computed.
            discretisation_option (str): specifies the "wrapper" method.
        """
        if discretisation_option == "recovered" and self.limiter is not None:
            self.x_brok_interpolator.interpolate()
        self.x_out_projector.project()
        x_out.assign(self.x_projected)

    @property
    def nlevels(self):
        return 1

    @abstractproperty
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x_out, self.idx),
            map_if_false=drop)

        return l.form

    @abstractproperty
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, self.idx))

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: -self.dt*t)

        return r.form

    def replace_transport_term(self):
        """
        Replaces a transport term with some other transport term.

        This routine allows the default transport term to be replaced with a
        different one, specified through the transport options. This is
        necessary because when the prognostic equations are declared,
        the particular transport scheme is not known. The details of the new
        transport term are stored in the time discretisation's options object.
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
        """
        Replace the transport velocity.

        Args:
            uadv (:class:`ufl.Expr`): the new transporting velocity.
        """
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
        """Set up the problem and the solver."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @abstractmethod
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        pass


class ExplicitTimeDiscretisation(TimeDiscretisation):
    """Base class for explicit time discretisations."""

    def __init__(self, state, field_name=None, subcycles=None,
                 solver_parameters=None, limiter=None, options=None):
        """
        Args:
            state (:class:`State`): the model's state object.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycles (int, optional): the number of sub-steps to perform.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        super().__init__(state, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)

        self.subcycles = subcycles

    def setup(self, equation, uadv, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            uadv (:class:`ufl.Expr`, optional): the transporting velocity.
                Defaults to None.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        super().setup(equation, uadv, *active_labels)

        # if user has specified a number of subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
        if self.subcycles is not None:
            self.dt = self.dt/self.subcycles
            self.ncycles = self.subcycles
        else:
            self.dt = self.dt
            self.ncycles = 1
        self.x = []
        for i in range(self.ncycles+1):
            self.x.append(Function(self.fs))

    @abstractmethod
    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        pass

    @embedded_dg
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        #print("before in", [(i, self.x[i].dat.data.min(), self.x[i].dat.data.max()) for i in range(self.ncycles+1)])
        #print(self.x[0])
        #print(self.x[1])
        self.x[0].assign(x_in)
        #print("in", [(i, self.x[i].dat.data.min(), self.x[i].dat.data.max()) for i in range(self.ncycles+1)])
        for i in range(self.ncycles):
            #print("before i", i, self.x[i].dat.data.min(), self.x[i].dat.data.max())
            #print("before i+1", i+1, self.x[i+1].dat.data.min(), self.x[i+1].dat.data.max())
            self.apply_cycle(self.x[i+1], self.x[i])
            print("after i", i, self.x[i].dat.data.min(), self.x[i].dat.data.max())
            print("after i+1", i+1, self.x[i+1].dat.data.min(), self.x[i+1].dat.data.max())
            self.x[i].assign(self.x[i+1])
        x_out.assign(self.x[self.ncycles])
        #print(self.ncycles, self.ncycles-1)
        print("out", x_out.dat.data.min(), x_out.dat.data.max())


class ForwardEuler(ExplicitTimeDiscretisation):
    """
    Implements the forward Euler timestepping scheme.

    The forward Euler method for operator F is the most simple explicit scheme:
    y^(n+1) = y^n + dt*F[y^n].
    """

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(ForwardEuler, self).lhs

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        return super(ForwardEuler, self).rhs

    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        self.x1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.x_out)


class SSPRK3(ExplicitTimeDiscretisation):
    u"""
    Implements the 3-stage Strong-Stability-Prevering Runge-Kutta method.

    The 3-stage Strong-Stability-Preserving Runge-Kutta method (SSPRK), for
    solving ∂y/∂t = F(y). It can be written as:

    y_1 = y^n + F[y^n]
    y_2 = (3/4)y^n + (1/4)(y_1 + F[y_1])
    y^(n+1) = (1/3)y^n + (2/3)(y_2 + F[y_2])

    where superscripts indicate the time-level and subscripts indicate the stage
    number. See e.g. Shu and Osher (1988).
    """

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(SSPRK3, self).lhs

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        return super(SSPRK3, self).rhs

    def solve_stage(self, x_in, stage):
        """
        Perform a single stage of the Runge-Kutta scheme.

        Args:
            x_in (:class:`Function`): field at the start of the stage.
            stage (int): index of the stage.
        """
        if stage == 0:
            self.solver.solve()
            self.x1.assign(self.x_out)

        elif stage == 1:
            self.solver.solve()
            self.x1.assign(0.75*x_in + 0.25*self.x_out)

        elif stage == 2:
            self.solver.solve()
            self.x1.assign((1./3.)*x_in + (2./3.)*self.x_out)

        if self.limiter is not None:
            self.limiter.apply(self.x1)

    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        if self.limiter is not None:
            self.limiter.apply(x_in)

        self.x1.assign(x_in)
        for i in range(3):
            self.solve_stage(x_in, i)
        x_out.assign(self.x1)


class RK4(ExplicitTimeDiscretisation):
    u"""
    Implements the classic 4-stage Runge-Kutta method.

    The classic 4-stage Runge-Kutta method for solving ∂y/∂t = F(y). It can be
    written as:

    k1 = F[y^n]
    k2 = F[y^n + 1/2*dt*k1]
    k3 = F[y^n + 1/2*dt*k2]
    k4 = F[y^n + dt*k3]
    y^(n+1) = y^n + (1/6) * dt * (k1 + 2*k2 + 2*k3 + k4)

    where superscripts indicate the time-level.
    """

    def setup(self, equation, uadv, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            uadv (:class:`ufl.Expr`, optional): the transporting velocity.
                Defaults to None.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        super().setup(equation, uadv, *active_labels)

        self.k1 = Function(self.fs)
        self.k2 = Function(self.fs)
        self.k3 = Function(self.fs)
        self.k4 = Function(self.fs)

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x_out, self.idx),
            map_if_false=drop)

        return l.form

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, self.idx))

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=drop,
            map_if_false=lambda t: -1*t)

        return r.form

    def solve_stage(self, x_in, stage):
        """
        Perform a single stage of the Runge-Kutta scheme.

        Args:
            x_in (:class:`Function`): field at the start of the stage.
            stage (int): index of the stage.
        """
        if stage == 0:
            self.solver.solve()
            self.k1.assign(self.x_out)
            self.x1.assign(x_in + 0.5 * self.dt * self.k1)

        elif stage == 1:
            self.solver.solve()
            self.k2.assign(self.x_out)
            self.x1.assign(x_in + 0.5 * self.dt * self.k2)

        elif stage == 2:
            self.solver.solve()
            self.k3.assign(self.x_out)
            self.x1.assign(x_in + self.dt * self.k3)

        elif stage == 3:
            self.solver.solve()
            self.k4.assign(self.x_out)
            self.x1.assign(x_in + 1/6 * self.dt * (self.k1 + 2*self.k2 + 2*self.k3 + self.k4))

    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        if self.limiter is not None:
            self.limiter.apply(x_in)

        self.x1.assign(x_in)

        for i in range(4):
            self.solve_stage(x_in, i)
        x_out.assign(self.x1)


class Heun(ExplicitTimeDiscretisation):
    u"""
    Implements Heun's method.

    The 2-stage Runge-Kutta scheme known as Heun's method,for solving
    ∂y/∂t = F(y). It can be written as:

    y_1 = F[y^n]
    y^(n+1) = (1/2)y^n + (1/2)F[y_1]

    where superscripts indicate the time-level and subscripts indicate the stage
    number.
    """
    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(Heun, self).lhs

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        return super(Heun, self).rhs

    def solve_stage(self, x_in, stage):
        """
        Perform a single stage of the Runge-Kutta scheme.

        Args:
            x_in (:class:`Function`): field at the start of the stage.
            stage (int): index of the stage.
        """
        if stage == 0:
            self.solver.solve()
            self.x1.assign(self.x_out)

        elif stage == 1:
            self.solver.solve()
            self.x1.assign(0.5 * x_in + 0.5 * (self.x_out))

    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        if self.limiter is not None:
            self.limiter.apply(x_in)

        self.x1.assign(x_in)
        for i in range(2):
            self.solve_stage(x_in, i)
        x_out.assign(self.x1)


class BackwardEuler(TimeDiscretisation):
    """
    Implements the backward Euler timestepping scheme.

    The backward Euler method for operator F is the most simple implicit scheme:
    y^(n+1) = y^n + dt*F[y^(n+1)].
    """
    def __init__(self, state, field_name=None, solver_parameters=None,
                 limiter=None, options=None):
        if isinstance(options, (EmbeddedDGOptions, RecoveredOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        super().__init__(state=state, field_name=field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)

    @property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.dt*t)

        return l.form

    @property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x1, self.idx),
            map_if_false=drop)

        return r.form

    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        self.x1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.x_out)


class ThetaMethod(TimeDiscretisation):
    """
    Implements the theta implicit-explicit timestepping method.

    The theta implicit-explicit timestepping method for operator F is written as
    y^(n+1) = y^n + dt*(1-theta)*F[y^n] + dt*theta*F[y^(n+1)]
    for off-centring parameter theta.
    """

    def __init__(self, state, field_name=None, theta=None,
                 solver_parameters=None, options=None):
        """
        Args:
            state (:class:`State`): the model's state object.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            theta (float, optional): the off-centring parameter. theta = 1
                corresponds to a backward Euler method. Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.

        Raises:
            ValueError: if theta is not provided.
        """
        # TODO: would this be better as a non-optional argument? Or should the
        # check be on the provided value?
        if theta is None:
            raise ValueError("please provide a value for theta between 0 and 1")
        if isinstance(options, (EmbeddedDGOptions, RecoveredOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
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
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.theta*self.dt*t)

        return l.form

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -(1-self.theta)*self.dt*t)

        return r.form

    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        self.x1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.x_out)


class ImplicitMidpoint(ThetaMethod):
    """
    Implements the implicit midpoint timestepping method.

    The implicit midpoint timestepping method for operator F is written as
    y^(n+1) = y^n + dt/2*F[y^n] + dt/2*F[y^(n+1)].
    It is equivalent to the "theta" method with theta = 1/2.
    """

    def __init__(self, state, field_name=None, solver_parameters=None,
                 options=None):
        """
        Args:
            state (:class:`State`): the model's state object.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        super().__init__(state, field_name, theta=0.5,
                         solver_parameters=solver_parameters,
                         options=options)


class MultilevelTimeDiscretisation(TimeDiscretisation):

    def __init__(self, state, field_name=None, solver_parameters=None,
                 limiter=None, options=None):
        if isinstance(options, (EmbeddedDGOptions, RecoveredOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        super().__init__(state=state, field_name=field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)
        self.initial_timesteps = 0

    @abstractproperty
    def nlevels(self):
        pass

    def setup(self, equation, uadv=None, apply_bcs=True, *active_labels):
        super().setup(equation=equation, uadv=uadv, apply_bcs=apply_bcs,
                      *active_labels)
        for n in range(self.nlevels, 1, -1):
            setattr(self, "xnm%i" % (n-1), Function(self.fs))


class BDF2(MultilevelTimeDiscretisation):

    @property
    def nlevels(self):
        return 2

    @property
    def lhs0(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.dt*t)

        return l.form

    @property
    def rhs0(self):
        """Set up the time discretisation's right hand side."""
        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x1, self.idx),
            map_if_false=drop)

        return r.form

    @property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: (2/3)*self.dt*t)

        return l.form

    @property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        xn = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x1, self.idx),
            map_if_false=drop)
        xnm1 = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.xnm1, self.idx),
            map_if_false=drop)

        r = (4/3.) * xn - (1/3.) * xnm1

        return r.form

    @property
    def solver0(self):
        """Set up the problem and the solver."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs0-self.rhs0, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"0"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @property
    def solver(self):
        """Set up the problem and the solver."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    def apply(self, x_out, *x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        if self.initial_timesteps < self.nlevels-1:
            self.initial_timesteps += 1
            solver = self.solver0
        else:
            solver = self.solver

        self.xnm1.assign(x_in[0])
        self.x1.assign(x_in[1])
        solver.solve()
        x_out.assign(self.x_out)
