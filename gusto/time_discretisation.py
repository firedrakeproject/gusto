u"""
Objects for discretising time derivatives.

Time discretisation objects discretise ∂y/∂t = F(y), for variable y, time t and
operator F.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import (Function, NonlinearVariationalProblem, split,
                       NonlinearVariationalSolver, Projector, Interpolator,
                       BrokenElement, VectorElement, FunctionSpace,
                       TestFunction, Constant, dot, grad, as_ufl,
                       DirichletBC)
from firedrake.formmanipulation import split_form
from firedrake.utils import cached_property
import ufl
from gusto.configuration import (logger, DEBUG, TransportEquationType,
                                 EmbeddedDGOptions, RecoveryOptions)
from gusto.labels import (time_derivative, transporting_velocity, prognostic,
                          subject, physics, transport, ibp_label, nonlinear,
                          replace_subject, replace_test_function)
from gusto.recovery import Recoverer, ReversibleRecoverer
from gusto.fml.form_manipulation_labelling import Term, all_terms, drop
from gusto.transport_forms import advection_form, continuity_form


__all__ = ["ForwardEuler", "BackwardEuler", "SSPRK3", "RK4", "Heun",
           "ThetaMethod", "ImplicitMidpoint", "BDF2", "TR_BDF2", "Leapfrog", "AdamsMoulton", "AdamsBashforth"]


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

    def __init__(self, domain, field_name=None, solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
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
        self.domain = domain
        self.field_name = field_name
        self.equation = None

        self.dt = domain.dt

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
        self.equation = equation
        self.residual = equation.residual

        if self.field_name is not None and hasattr(equation, "field_names"):
            self.idx = equation.field_names.index(self.field_name)
            self.fs = equation.spaces[self.idx]
            self.residual = self.residual.label_map(
                lambda t: t.get(prognostic) == self.field_name,
                lambda t: Term(
                    split_form(t.form)[self.idx].form,
                    t.labels),
                drop)

        else:
            self.field_name = equation.field_name
            self.fs = equation.function_space
            self.idx = None

        bcs = equation.bcs[self.field_name]

        if len(active_labels) > 0:
            self.residual = self.residual.label_map(
                lambda t: any(t.has_label(time_derivative, *active_labels)),
                map_if_false=drop)

        if nonlinear in active_labels:
            self.residual = self.residual.label_map(
                lambda t: t.has_label(nonlinear),
                map_if_true=lambda t: Term(t.get(nonlinear).form, t.labels))

        self.evaluate_source = []
        for t in self.residual:
            if t.has_label(physics):
                self.evaluate_source.append(t.get(physics))

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
                self.fs = FunctionSpace(self.domain.mesh, V_elt)
            else:
                self.fs = options.embedding_space
            self.xdg_in = Function(self.fs)
            self.xdg_out = Function(self.fs)
            if self.idx is None:
                self.x_projected = Function(equation.function_space)
            else:
                self.x_projected = Function(equation.spaces[self.idx])
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
            dim = self.domain.mesh.topological_dimension()
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
            self.interp_back = False
            if self.limiter is None:
                self.x_out_projector = Projector(self.xdg_out, self.x_projected,
                                                 solver_parameters=parameters)
            else:
                self.x_out_projector = Recoverer(self.xdg_out, self.x_projected)

        if self.discretisation_option == "recovered":
            # set up the necessary functions
            if self.idx is not None:
                self.x_in = Function(equation.spaces[self.idx])
            else:
                self.x_in = Function(equation.function_space)

            # Operator to recover to higher discontinuous space
            self.x_recoverer = ReversibleRecoverer(self.x_in, self.xdg_in, options)

            self.interp_back = (options.project_low_method == 'interpolate')
            if options.project_low_method == 'interpolate':
                self.x_out_projector = Interpolator(self.xdg_out, self.x_projected)
            elif options.project_low_method == 'project':
                self.x_out_projector = Projector(self.xdg_out, self.x_projected)
            elif options.project_low_method == 'recover':
                self.x_out_projector = Recoverer(self.xdg_out, self.x_projected, method=options.broken_method)

            if self.limiter is not None and options.project_low_method != 'recover':
                logger.warning('A limiter has been requested for a recovered transport scheme, but the method for projecting back is not recovery')

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
            self.x_recoverer.project()

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
        self.x_out_projector.interpolate() if self.interp_back else self.x_out_projector.project()
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

        for t in l:
            print(t.form)
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

        for t in r:
            print(t.form)
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
                if self.idx is not None:
                    field = self.equation.X.split()[self.idx]
                else:
                    field = self.equation.X
                test = TestFunction(self.fs)

                # Set up new transport term (depending on the type of transport equation)
                if old_transport_term.labels['transport'] == TransportEquationType.advective:
                    new_transport_term = advection_form(self.domain, test, field, ibp=self.options.ibp)
                elif old_transport_term.labels['transport'] == TransportEquationType.conservative:
                    new_transport_term = continuity_form(self.domain, test, field, ibp=self.options.ibp)
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
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        pass


class ExplicitTimeDiscretisation(TimeDiscretisation):
    """Base class for explicit time discretisations."""

    def __init__(self, domain, field_name=None, subcycles=None,
                 solver_parameters=None, limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
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
        super().__init__(domain, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)

        self.subcycles = subcycles

    def setup(self, equation, uadv, apply_bcs=True, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            uadv (:class:`ufl.Expr`, optional): the transporting velocity.
                Defaults to None.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        super().setup(equation, uadv, apply_bcs, *active_labels)

        # if user has specified a number of subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
        if self.subcycles is not None:
            self.dt = self.dt/self.subcycles
            self.ncycles = self.subcycles
        else:
            self.dt = self.dt
            self.ncycles = 1
        self.x0 = Function(self.fs)
        self.x1 = Function(self.fs)

    @abstractmethod
    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        pass

    @embedded_dg
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        self.x0.assign(x_in)
        for i in range(self.ncycles):
            for evaluate in self.evaluate_source:
                evaluate(x_in, self.dt)
            self.apply_cycle(self.x1, self.x0)
            self.x0.assign(self.x1)
        x_out.assign(self.x1)


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
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
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
    def __init__(self, domain, field_name=None, solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
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
                to control the "wrapper" methods. Defaults to None.

        Raises:
            NotImplementedError: if options is an instance of
            EmbeddedDGOptions or RecoveryOptions
        """
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        super().__init__(domain=domain, field_name=field_name,
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
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
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

    def __init__(self, domain, field_name=None, theta=None,
                 solver_parameters=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
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
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super().__init__(domain, field_name,
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
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
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

    def __init__(self, domain, field_name=None, solver_parameters=None,
                 options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        super().__init__(domain, field_name, theta=0.5,
                         solver_parameters=solver_parameters,
                         options=options)


class MultilevelTimeDiscretisation(TimeDiscretisation):
    """Base class for multi-level timesteppers"""

    def __init__(self, domain, field_name=None, solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
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
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        super().__init__(domain=domain, field_name=field_name,
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
    """
    Implements the implicit multistep BDF2 timestepping method

    The BDF2 timestepping method for operator F is written as
    y^(n+1) = (4/3)*y^n - (1/3)*y^(n-1) + (2/3)*dt*F[y^(n+1)]
    """

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
        """Set up the time discretisation's right hand side for inital BDF step."""
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
        """Set up the time discretisation's right hand side for BDF2 steps."""
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
        """Set up the problem and the solver for initial BDF step."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs0-self.rhs0, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"0"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @property
    def solver(self):
        """Set up the problem and the solver for BDF2 steps."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    def apply(self, x_out, *x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
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


class TR_BDF2(TimeDiscretisation):
    """
    Implements the two stage implicit TR-BDF2 time stepping method, with a trapezoidal stage (TR) followed
    by a second order backwards difference stage (BDF2).

    The TR-BDF2 time stepping method for operator F is written as
    y^(n+g) = y^n + dt*g/2*F[y^n] + dt*g/2*F[y^(n+g)] (TR stage)
    y^(n+1) = 1/(g(2-g))*y^(n+g) - (1-g)**2/(g(2-g))*y^(n) + (1-g)/(2-g)*dt*F[y^(n+1)] (BDF2 stage)
    for an off-centring parameter g (gamma).
    """
    def __init__(self, domain, gamma, field_name=None,
                 solver_parameters=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            gamma (float): the off-centring parameter
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        if (gamma < 0. or gamma > 1.):
            raise ValueError("please provide a value for gamma between 0 and 1")
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super().__init__(domain, field_name,
                         solver_parameters=solver_parameters,
                         options=options)

        self.gamma = gamma

    def setup(self, equation, uadv=None, apply_bcs=True, *active_labels):
        super().setup(equation, uadv, apply_bcs, *active_labels)
        self.xnpg = Function(self.fs)
        self.xn = Function(self.fs)

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative) for the TR stage."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.xnpg, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: 0.5*self.gamma*self.dt*t)

        return l.form

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side for the TR stage."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.xn, self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -0.5*self.gamma*self.dt*t)

        return r.form

    @cached_property
    def lhs_bdf2(self):
        """Set up the discretisation's left hand side (the time derivative) for the BDF2 stage."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: ((1.0-self.gamma)/(2.0-self.gamma))*self.dt*t)

        return l.form

    @cached_property
    def rhs_bdf2(self):
        """Set up the time discretisation's right hand side for the BDF2 stage."""
        xn = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.xn, self.idx),
            map_if_false=drop)
        xnpg = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.xnpg, self.idx),
            map_if_false=drop)

        r = (1.0/(self.gamma*(2.0-self.gamma)))*xnpg - ((1.0-self.gamma)**2/(self.gamma*(2.0-self.gamma))) * xn

        return r.form

    @cached_property
    def solver_tr(self):
        """Set up the problem and the solver."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.xnpg, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_tr"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @cached_property
    def solver_bdf2(self):
        """Set up the problem and the solver."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs_bdf2-self.rhs_bdf2, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_bdf2"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
        """
        self.xn.assign(x_in)
        self.solver_tr.solve()
        self.solver_bdf2.solve()
        x_out.assign(self.x_out)


class Leapfrog(MultilevelTimeDiscretisation):
    """
    Implements the multistep Leapfrog timestepping method.

    The Leapfrog timestepping method for operator F is written as
    y^(n+1) = y^(n-1)  + 2*dt*F[y^n]
    """
    @property
    def nlevels(self):
        return 2

    @property
    def rhs0(self):
        """Set up the discretisation's right hand side for initial forward euler step."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -self.dt*t)

        return r.form

    @property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(Leapfrog, self).lhs

    @property
    def rhs(self):
        """Set up the discretisation's right hand side for leapfrog steps."""
        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=replace_subject(self.x1, self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_true=replace_subject(self.xnm1, self.idx),
                        map_if_false=lambda t: -2.0*self.dt*t)

        return r.form

    @property
    def solver0(self):
        """Set up the problem and the solver for initial forward euler step."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs0, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"0"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @property
    def solver(self):
        """Set up the problem and the solver for leapfrog steps."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    def apply(self, x_out, *x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
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


class AdamsBashforth(MultilevelTimeDiscretisation):
    """
    Implements the explicit multistep Adams-Bashforth timestepping method of general order up to 5

    The general AB timestepping method for operator F is written as
    y^(n+1) = y^n + dt*(b_0*F[y^(n)] + b_1*F[y^(n-1)] + b_2*F[y^(n-2)] + b_3*F[y^(n-3)] + b_4*F[y^(n-4)])
    """
    def __init__(self, domain, order, field_name=None,
                 solver_parameters=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            order (float, optional): order of scheme
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.

        Raises:
            ValueError: if order is not provided, or is in incorrect range.
        """

        if (order > 5 or order < 1):
            raise ValueError("Adams-Bashforth of order greater than 5 not implemented")
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")

        super().__init__(domain, field_name,
                         solver_parameters=solver_parameters,
                         options=options)

        self.order = order

    def setup(self, equation, uadv=None, apply_bcs=True, *active_labels):
        super().setup(equation=equation, uadv=uadv, apply_bcs=apply_bcs,
                      *active_labels)

        self.x = [Function(self.fs) for i in range(self.nlevels)]

        if (self.order == 1):
            self.b = [1.0]
        elif (self.order == 2):
            self.b = [-(1.0/2.0), (3.0/2.0)]
        elif (self.order == 3):
            self.b = [(5.0)/(12.0), -(16.0)/(12.0), (23.0)/(12.0)]
        elif (self.order == 4):
            self.b = [-(9.0)/(24.0), (37.0)/(24.0), -(59.0)/(24.0), (55.0)/(24.0)]
        elif (self.order == 5):
            self.b = [(251.0)/(720.0), -(1274.0)/(720.0), (2616.0)/(720.0), -(2774.0)/(720.0), (2901.0)/(720.0)]

    @property
    def nlevels(self):
        return self.order

    @property
    def rhs0(self):
        """Set up the discretisation's right hand side for initial forward euler step."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x[-1], self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -self.dt*t)

        return r.form

    @property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(AdamsBashforth, self).lhs

    @property
    def rhs(self):
        """Set up the discretisation's right hand side for Adams Bashforth steps."""
        r = self.residual.label_map(all_terms,
                                    map_if_true=replace_subject(self.x[-1], self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -self.b[-1]*self.dt*t)
        for n in range(self.nlevels-1):
            rtemp = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                            map_if_true=drop,
                                            map_if_false=replace_subject(self.x[n], self.idx))
            rtemp = rtemp.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: -self.dt*self.b[n]*t)
            r += rtemp
        return r.form

    @property
    def solver0(self):
        """Set up the problem and the solverfor initial forward euler step."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs0, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"0"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @property
    def solver(self):
        """Set up the problem and the solver for Adams Bashforth steps."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    def apply(self, x_out, *x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
        """
        if self.initial_timesteps < self.nlevels-1:
            self.initial_timesteps += 1
            solver = self.solver0
        else:
            solver = self.solver

        for n in range(self.nlevels):
            self.x[n].assign(x_in[n])
        solver.solve()
        x_out.assign(self.x_out)


class AdamsMoulton(MultilevelTimeDiscretisation):
    """
    Implements the implicit multistep Adams-Moulton timestepping method of general order up to 5

    The general AM timestepping method for operator F is written as
    y^(n+1) = y^n + dt*(b_0*F[y^(n+1)] + b_1*F[y^(n)] + b_2*F[y^(n-1)] + b_3*F[y^(n-2)])
    """
    def __init__(self, domain, order, field_name=None,
                 solver_parameters=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            order (float, optional): order of scheme
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.

        Raises:
            ValueError: if order is not provided, or is in incorrect range.
        """
        if (order > 4 or order < 1):
            raise ValueError("Adams-Moulton of order greater than 5 not implemented")
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        if not solver_parameters:
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super().__init__(domain, field_name,
                         solver_parameters=solver_parameters,
                         options=options)

        self.order = order

    def setup(self, equation, uadv=None, apply_bcs=True, *active_labels):
        super().setup(equation=equation, uadv=uadv, apply_bcs=apply_bcs,
                      *active_labels)

        self.x = [Function(self.fs) for i in range(self.nlevels)]

        if (self.order == 1):
            self.bl = (1.0/2.0)
            self.br = [(1.0/2.0)]
        elif (self.order == 2):
            self.bl = (5.0/12.0)
            self.br = [-(1.0/12.0), (8.0/12.0)]
        elif (self.order == 3):
            self.bl = (9.0/24.0)
            self.br = [(1.0/24.0), -(5.0/24.0), (19.0/24.0)]
        elif (self.order == 4):
            self.bl = (251.0/720.0)
            self.br = [-(19.0/720.0), (106.0/720.0), -(254.0/720.0), (646.0/720.0)]

    @property
    def nlevels(self):
        return self.order

    @property
    def rhs0(self):
        """Set up the discretisation's right hand side for initial trapezoidal step."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x[-1], self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -0.5*self.dt*t)

        return r.form

    @property
    def lhs0(self):
        """Set up the time discretisation's right hand side for initial trapezoidal step."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: 0.5*self.dt*t)
        return l.form

    @property
    def lhs(self):
        """Set up the time discretisation's right hand side for Adams Moulton steps."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.bl*self.dt*t)
        return l.form

    @property
    def rhs(self):
        """Set up the discretisation's right hand side for Adams Moulton steps."""
        r = self.residual.label_map(all_terms,
                                    map_if_true=replace_subject(self.x[-1], self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -self.br[-1]*self.dt*t)
        for n in range(self.nlevels-1):
            rtemp = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                            map_if_true=drop,
                                            map_if_false=replace_subject(self.x[n], self.idx))
            rtemp = rtemp.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: -self.dt*self.br[n]*t)
            r += rtemp
        return r.form

    @property
    def solver0(self):
        """Set up the problem and the solver for initial trapezoidal step."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs0-self.rhs0, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"0"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @property
    def solver(self):
        """Set up the problem and the solver for Adams Moulton steps."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    def apply(self, x_out, *x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
        """
        if self.initial_timesteps < self.nlevels-1:
            self.initial_timesteps += 1
            print(self.initial_timesteps)
            solver = self.solver0
        else:
            solver = self.solver

        for n in range(self.nlevels):
            self.x[n].assign(x_in[n])
        solver.solve()
        x_out.assign(self.x_out)
