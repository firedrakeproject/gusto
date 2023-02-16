"""Classes for controlling the timestepping loop."""

from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import Function, Projector, Constant, Mesh
from pyop2.profiling import timed_stage
from gusto.configuration import logger
from gusto.equations import PrognosticEquationSet
from gusto.forcing import Forcing
from gusto.fml.form_manipulation_labelling import drop, Label
from gusto.labels import (transport, diffusion, time_derivative,
                          linearisation, prognostic, physics)
from gusto.linear_solvers import LinearTimesteppingSolver
from gusto.fields import TimeLevelFields, StateFields
from gusto.time_discretisation import ExplicitTimeDiscretisation

__all__ = ["Timestepper", "SplitPhysicsTimestepper", "SemiImplicitQuasiNewton",
           "PrescribedTransport", "MeshMovement"]


class BaseTimestepper(object, metaclass=ABCMeta):
    """Base class for timesteppers."""

    def __init__(self, equation, io):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation.
            io (:class:`IO`): the model's object for controlling input/output.
        """

        self.equation = equation
        self.io = io
        self.dt = self.equation.domain.dt
        self.t = Constant(0.0)
        self.reference_profiles_initialised = False

        self.setup_fields()
        self.setup_scheme()

        self.io.log_parameters(equation)
        self.io.setup_diagnostics(self.fields)

    @abstractproperty
    def transporting_velocity(self):
        return NotImplementedError

    @abstractmethod
    def setup_fields(self):
        """Set up required fields. Must be implemented in child classes"""
        pass

    @abstractmethod
    def setup_scheme(self):
        """Set up required scheme(s). Must be implemented in child classes"""
        pass

    @abstractmethod
    def timestep(self):
        """Defines the timestep. Must be implemented in child classes"""
        return NotImplementedError

    def run(self, t, tmax, pickup=False):
        """
        Runs the model for the specified time, from t to tmax

        Args:
            t (float): the start time of the run
            tmax (float): the end time of the run
            pickup: (bool): specify whether to pickup from a previous run
        """

        if pickup:
            t = self.io.pickup_from_checkpoint(self.fields)

        with timed_stage("Dump output"):
            self.io.setup_dump(self.fields, t, tmax, pickup)

        self.t.assign(t)

        while float(self.t) < tmax - 0.5*float(self.dt):
            logger.info(f'at start of timestep, t={float(self.t)}, dt={float(self.dt)}')

            self.x.update()

            self.timestep()

            self.t.assign(self.t + self.dt)

            with timed_stage("Dump output"):
                self.io.dump(self.fields, float(self.t))

        if self.io.output.checkpoint:
            self.io.chkpt.close()

        logger.info(f'TIMELOOP complete. t={float(self.t)}, tmax={tmax}')

    def set_reference_profiles(self, reference_profiles):
        """
        Initialise the model's reference profiles.

        reference_profiles (list): an iterable of pairs: (field_name, expr),
            where 'field_name' is the string giving the name of the reference
            profile field expr is the :class:`ufl.Expr` whose value is used to
            set the reference field.
        """
        for field_name, profile in reference_profiles:
            if field_name+'_bar' in self.fields:
                # For reference profiles already added to state, allow
                # interpolation from expressions
                ref = self.fields(field_name+'_bar')
            elif isinstance(profile, Function):
                # Need to add reference profile to state so profile must be
                # a Function
                ref = self.fields(field_name+'_bar', space=profile.function_space(), dump=False)
            else:
                raise ValueError(f'When initialising reference profile {field_name}'
                                 + ' the passed profile must be a Function')
            ref.interpolate(profile)

            # Assign profile to X_ref belonging to equation
            if isinstance(self.equation, PrognosticEquationSet):
                assert field_name in self.equation.field_names, \
                    f'Cannot set reference profile as field {field_name} not found'
                idx = self.equation.field_names.index(field_name)
                X_ref = self.equation.X_ref.split()[idx]
                X_ref.assign(ref)

        self.reference_profiles_initialised = True


class Timestepper(BaseTimestepper):
    """
    Implements a timeloop by applying a scheme to a prognostic equation.
    """

    def __init__(self, equation, scheme, io):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation
            io (:class:`IO`): the model's object for controlling input/output.
        """
        self.scheme = scheme
        super().__init__(equation=equation, io=io)

    @property
    def transporting_velocity(self):
        return "prognostic"

    def setup_fields(self):
        self.x = TimeLevelFields(self.equation, self.scheme.nlevels)
        self.fields = StateFields(self.x.np1, self.equation.prescribed_fields,
                                  *self.io.output.dumplist)

    def setup_scheme(self):
        self.scheme.setup(self.equation, self.transporting_velocity)

    def timestep(self):
        """
        Implement the timestep
        """
        xnp1 = self.x.np1
        name = self.equation.field_name
        x_in = [x(name) for x in self.x.previous[-self.scheme.nlevels:]]

        self.scheme.apply(xnp1(name), *x_in)


class SplitPhysicsTimestepper(Timestepper):
    """
    Implements a timeloop by applying schemes separately to the physics and
    dynamics. This 'splits' the physics from the dynamics and allows a different
    scheme to be applied to the physics terms than the prognostic equation.
    """

    def __init__(self, equation, scheme, io, physics_schemes=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation
            io (:class:`IO`): the model's object for controlling input/output.
            physics_schemes: (list, optional): a list of :class:`Physics` and
                :class:`TimeDiscretisation` options describing physical
                parametrisations and timestepping schemes to use for each.
                Timestepping schemes for physics must be explicit. Defaults to
                None.
        """

        super().__init__(equation, scheme, io)

        if physics_schemes is not None:
            self.physics_schemes = physics_schemes
        else:
            self.physics_schemes = []

        for _, phys_scheme in self.physics_schemes:
            # check that the supplied schemes for physics are explicit
            assert isinstance(phys_scheme, ExplicitTimeDiscretisation), "Only explicit schemes can be used for physics"
            apply_bcs = False
            phys_scheme.setup(equation, self.transporting_velocity, apply_bcs, physics)

    @property
    def transporting_velocity(self):
        return "prognostic"

    def setup_scheme(self):
        # Go through and label all non-physics terms with a "dynamics" label
        dynamics = Label('dynamics')
        self.equation.label_terms(lambda t: not any(t.has_label(time_derivative, physics)), dynamics)
        apply_bcs = True
        self.scheme.setup(self.equation, self.transporting_velocity, apply_bcs, dynamics)

    def timestep(self):

        super().timestep()

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))


class SemiImplicitQuasiNewton(BaseTimestepper):
    """
    Implements a semi-implicit quasi-Newton discretisation,
    with Strang splitting and auxiliary semi-Lagrangian transport.

    The timestep consists of an outer loop applying the transport and an
    inner loop to perform the quasi-Newton interations for the fast-wave
    terms.
    """

    def __init__(self, equation_set, io, transport_schemes,
                 auxiliary_equations_and_schemes=None,
                 linear_solver=None,
                 diffusion_schemes=None,
                 physics_schemes=None, **kwargs):

        """
        Args:
            equation_set (:class:`PrognosticEquationSet`): the prognostic
                equation set to be solved
            io (:class:`IO`): the model's object for controlling input/output.
            transport_schemes: iterable of ``(field_name, scheme)`` pairs
                indicating the name of the field (str) to transport, and the
                :class:`TimeDiscretisation` to use
            auxiliary_equations_and_schemes: iterable of ``(equation, scheme)``
                pairs indicating any additional equations to be solved and the
                scheme to use to solve them. Defaults to None.
            linear_solver: a :class:`.TimesteppingSolver` object. Defaults to
                None.
            diffusion_schemes: optional iterable of ``(field_name, scheme)``
                pairs indicating the fields to diffuse, and the
                :class:`~.Diffusion` to use. Defaults to None.
            physics_schemes: (list, optional): a list of :class:`Physics` and
                :class:`TimeDiscretisation` options describing physical
                parametrisations and timestepping schemes to use for each.
                Timestepping schemes for physics must be explicit. Defaults to
                None.

        :kwargs: maxk is the number of outer iterations, maxi is the number
            of inner iterations and alpha is the offcentering parameter
    """

        self.maxk = kwargs.pop("maxk", 4)
        self.maxi = kwargs.pop("maxi", 1)
        self.alpha = kwargs.pop("alpha", 0.5)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        if physics_schemes is not None:
            self.physics_schemes = physics_schemes
        else:
            self.physics_schemes = []
        for _, scheme in self.physics_schemes:
            assert scheme.nlevels == 1, "multilevel schemes not supported as part of this timestepping loop"
            assert isinstance(scheme, ExplicitTimeDiscretisation), "Only explicit schemes can be used for physics"

        self.active_transport = []
        for scheme in transport_schemes:
            assert scheme.nlevels == 1, "multilevel schemes not supported as part of this timestepping loop"
            assert scheme.field_name in equation_set.field_names
            self.active_transport.append((scheme.field_name, scheme))

        self.diffusion_schemes = []
        if diffusion_schemes is not None:
            for scheme in diffusion_schemes:
                assert scheme.nlevels == 1, "multilevel schemes not supported as part of this timestepping loop"
                assert scheme.field_name in equation_set.field_names
                self.diffusion_schemes.append((scheme.field_name, scheme))

        if auxiliary_equations_and_schemes is not None:
            for eqn, scheme in auxiliary_equations_and_schemes:
                assert not hasattr(eqn, "field_names"), 'Cannot use auxiliary schemes with multiple fields'
            self.auxiliary_schemes = [
                (eqn.field_name, scheme)
                for eqn, scheme in auxiliary_equations_and_schemes]
        else:
            auxiliary_equations_and_schemes = []
            self.auxiliary_schemes = []
        self.auxiliary_equations_and_schemes = auxiliary_equations_and_schemes

        super().__init__(equation_set, io)

        for aux_eqn, aux_scheme in self.auxiliary_equations_and_schemes:
            aux_scheme.setup(aux_eqn, self.transporting_velocity)

        self.tracers_to_copy = []
        for name in equation_set.field_names:
            # Extract time derivative for that prognostic
            mass_form = equation_set.residual.label_map(
                lambda t: (t.has_label(time_derivative) and t.get(prognostic) == name),
                map_if_false=drop)
            # Copy over field if the time derivative term has no linearisation
            if not mass_form.terms[0].has_label(linearisation):
                self.tracers_to_copy.append(name)

        self.field_name = equation_set.field_name
        W = equation_set.function_space
        self.xrhs = Function(W)
        self.dy = Function(W)
        if linear_solver is None:
            self.linear_solver = LinearTimesteppingSolver(equation_set, self.alpha)
        else:
            self.linear_solver = linear_solver
        self.forcing = Forcing(equation_set, self.alpha)
        self.bcs = equation_set.bcs

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.x.np1("u")

        for bc in self.bcs['u']:
            bc.apply(unp1)

    @property
    def transporting_velocity(self):
        """Computes ubar=(1-alpha)*un + alpha*unp1"""
        xn = self.x.n
        xnp1 = self.x.np1
        # computes ubar from un and unp1
        return xn('u') + self.alpha*(xnp1('u')-xn('u'))

    def setup_fields(self):
        """Sets up time levels n, star, p and np1"""
        self.x = TimeLevelFields(self.equation, 1)
        self.x.add_fields(self.equation, levels=("star", "p"))
        for aux_eqn, _ in self.auxiliary_equations_and_schemes:
            self.x.add_fields(aux_eqn)
        # Prescribed fields for auxiliary eqns should come from prognostics of
        # other equations, so only the prescribed fields of the main equation
        # need passing to StateFields
        self.fields = StateFields(self.x.np1, self.equation.prescribed_fields,
                                  *self.io.output.dumplist)

    def setup_scheme(self):
        """Sets up transport, diffusion and physics schemes"""
        # TODO: apply_bcs should be False for advection but this means
        # tests with KGOs fail
        apply_bcs = True
        for _, scheme in self.active_transport:
            scheme.setup(self.equation, self.transporting_velocity, apply_bcs, transport)
        apply_bcs = True
        for _, scheme in self.diffusion_schemes:
            scheme.setup(self.equation, self.transporting_velocity, apply_bcs, diffusion)
        for _, scheme in self.physics_schemes:
            apply_bcs = True
            scheme.setup(self.equation, self.transporting_velocity, apply_bcs, physics)

    def copy_active_tracers(self, x_in, x_out):
        """
        Copies active tracers from one set of fields to another, if those fields
        are not included in the linear solver. This is determined by whether the
        time derivative term for that tracer has a linearisation.

        Args:
           x_in:  The input set of fields
           x_out: The output set of fields
        """

        for name in self.tracers_to_copy:
            x_out(name).assign(x_in(name))

    def timestep(self):
        """Defines the timestep"""
        xn = self.x.n
        xnp1 = self.x.np1
        xstar = self.x.star
        xp = self.x.p
        xrhs = self.xrhs
        dy = self.dy

        with timed_stage("Apply forcing terms"):
            self.forcing.apply(xn, xn, xstar(self.field_name), "explicit")

        xp(self.field_name).assign(xstar(self.field_name))

        for k in range(self.maxk):

            with timed_stage("Transport"):
                for name, scheme in self.active_transport:
                    # transports a field from xstar and puts result in xp
                    scheme.apply(xp(name), xstar(name))

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(xp, xnp1, xrhs, "implicit")

                xrhs -= xnp1(self.field_name)

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve(xrhs, dy)  # solves linear system and places result in dy

                xnp1X = xnp1(self.field_name)
                xnp1X += dy

            # Update xnp1 values for active tracers not included in the linear solve
            self.copy_active_tracers(xp, xnp1)

            self._apply_bcs()

        for name, scheme in self.auxiliary_schemes:
            # transports a field from xn and puts result in xnp1
            scheme.apply(xnp1(name), xn(name))

        with timed_stage("Diffusion"):
            for name, scheme in self.diffusion_schemes:
                scheme.apply(xnp1(name), xnp1(name))

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(xnp1(scheme.field_name), xnp1(scheme.field_name))

    def run(self, t, tmax, pickup=False):
        """
        Runs the model for the specified time, from t to tmax.

        Args:
            t (float): the start time of the run
            tmax (float): the end time of the run
            pickup: (bool): specify whether to pickup from a previous run
        """

        assert self.reference_profiles_initialised, \
            'Reference profiles for must be initialised to use Semi-Implicit Timestepper'

        super().run(t, tmax, pickup=pickup)


class PrescribedTransport(Timestepper):
    """
    Implements a timeloop with a prescibed transporting velocity
    """
    def __init__(self, equation, scheme, io, physics_schemes=None,
                 prescribed_transporting_velocity=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation
            io (:class:`IO`): the model's object for controlling input/output.
            physics_schemes: (list, optional): a list of :class:`Physics` and
                :class:`TimeDiscretisation` options describing physical
                parametrisations and timestepping schemes to use for each.
                Timestepping schemes for physics must be explicit. Defaults to
                None.
            prescribed_transporting_velocity (func, optional): a function,
                with a single argument representing the time, that returns a
                :class:`ufl.Expr` for the transporting velocity. This allows
                the transporting velocity field to be updated with time. If
                `None` is provided then the equation's velocity field is not
                updated. Defaults to None.
        """

        super().__init__(equation, scheme, io)

        if physics_schemes is not None:
            self.physics_schemes = physics_schemes
        else:
            self.physics_schemes = []

        for _, scheme in self.physics_schemes:
            # check that the supplied schemes for physics are explicit
            assert isinstance(scheme, ExplicitTimeDiscretisation), "Only explicit schemes can be used for physics"
            apply_bcs = False
            scheme.setup(equation, self.transporting_velocity, apply_bcs, physics)

        if prescribed_transporting_velocity is not None:
            self.velocity_projection = Projector(
                prescribed_transporting_velocity(self.t),
                self.fields('u'))
        else:
            self.velocity_projection = None

    @property
    def transporting_velocity(self):
        return self.fields('u')

    def setup_fields(self):
        self.x = TimeLevelFields(self.equation, self.scheme.nlevels)
        self.fields = StateFields(self.x.np1, self.equation.prescribed_fields,
                                  *self.io.output.dumplist)

    def setup_scheme(self):
        self.scheme.setup(self.equation, self.transporting_velocity)

    def timestep(self):
        if self.velocity_projection is not None:
            self.velocity_projection.project()

        super().timestep()

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))


class MeshMovement(SemiImplicitQuasiNewton):

    def __init__(self, equation_set, io, transport_schemes,
                 auxiliary_equations_and_schemes=None,
                 linear_solver=None,
                 diffusion_schemes=None,
                 physics_schemes=None,
                 mesh_generator=None,
                 **kwargs):

        super().__init__(equation_set=equation_set,
                         io=io,
                         transport_schemes=transport_schemes,
                         auxiliary_equations_and_schemes=auxiliary_equations_and_schemes,
                         linear_solver=linear_solver,
                         diffusion_schemes=diffusion_schemes,
                         physics_schemes=physics_schemes,
                         **kwargs)
        self.mesh_generator = mesh_generator
        mesh = self.equation.domain.mesh
        self.X0 = Function(mesh.coordinates)
        self.X1 = Function(mesh.coordinates)
        xold = Function(mesh.coordinates)
        self.mesh_old = Mesh(xold)
        mesh_generator.monitor.setup(self.fields)
        mesh_generator.setup()

    def timestep(self):
        """Defines the timestep"""
        xn = self.x.n
        xnp1 = self.x.np1
        xstar = self.x.star
        xp = self.x.p
        xrhs = self.xrhs
        dy = self.dy

        with timed_stage("Apply forcing terms"):
            self.forcing.apply(xn, xn, xstar(self.field_name), "explicit")

        xp(self.field_name).assign(xstar(self.field_name))

        for k in range(self.maxk):

            # This is used as the "old mesh" domain in projections
            self.mesh_old.coordinates.dat.data[:] = self.X1.dat.data[:]
            self.X0.assign(self.X1)

            self.mesh_generator.pre_meshgen_callback()
            with timed_stage("Mesh generation"):
                self.X1.assign(self.mesh_generator.get_new_mesh())

            with timed_stage("Transport"):
                for name, scheme in self.active_transport:
                    # transports a field from xstar and puts result in xp
                    scheme.apply(xp(name), xstar(name))

            self.mesh_generator.post_meshgen_callback()

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(xp, xnp1, xrhs, "implicit")

                xrhs -= xnp1(self.field_name)

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve(xrhs, dy)  # solves linear system and places result in dy

                xnp1X = xnp1(self.field_name)
                xnp1X += dy

            # Update xnp1 values for active tracers not included in the linear solve
            self.copy_active_tracers(xp, xnp1)

            self._apply_bcs()

        for name, scheme in self.auxiliary_schemes:
            # transports a field from xn and puts result in xnp1
            scheme.apply(xnp1(name), xn(name))

        with timed_stage("Diffusion"):
            for name, scheme in self.diffusion_schemes:
                scheme.apply(xnp1(name), xnp1(name))

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(xnp1(scheme.field_name), xnp1(scheme.field_name))
