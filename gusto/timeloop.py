"""Classes for controlling the timestepping loop."""

from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import Function, Projector
from pyop2.profiling import timed_stage
from gusto.configuration import logger
from gusto.forcing import Forcing
from gusto.fml.form_manipulation_labelling import drop
from gusto.labels import (transport, diffusion, time_derivative,
                          linearisation, prognostic)
from gusto.linear_solvers import LinearTimesteppingSolver
from gusto.fields import TimeLevelFields

__all__ = ["Timestepper", "SemiImplicitQuasiNewton",
           "PrescribedTransport"]


class BaseTimestepper(object, metaclass=ABCMeta):
    """Base class for timesteppers."""

    def __init__(self, equation, state):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation.
            state (:class:`State`): the model's state object
        """

        self.equation = equation
        self.state = state

        self.setup_timeloop()

    @abstractproperty
    def transporting_velocity(self):
        return NotImplementedError

    def setup_timeloop(self):
        """Sets up the fields and scheme used in the timeloop"""
        self.setup_fields()
        self.setup_scheme()

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

        state = self.state

        if pickup:
            t = state.pickup_from_checkpoint()

        state.setup_diagnostics()

        with timed_stage("Dump output"):
            state.setup_dump(t, tmax, pickup)

        state.t.assign(t)

        self.x.initialise(state)

        while float(state.t) < tmax - 0.5*float(state.dt):
            logger.info(f'at start of timestep, t={float(state.t)}, dt={float(state.dt)}')

            self.x.update()

            self.timestep()

            for field in self.x.np1:
                state.fields(field.name()).assign(field)

            state.t.assign(state.t + state.dt)

            with timed_stage("Dump output"):
                state.dump(float(state.t))

        if state.output.checkpoint:
            state.chkpt.close()

        logger.info(f'TIMELOOP complete. t={float(state.t)}, tmax={tmax}')


class Timestepper(BaseTimestepper):
    """
    Implements a timeloop by applying a scheme to a prognostic equation.
    """

    def __init__(self, equation, scheme, state):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation
            state (:class:`State`): the model's state object
        """
        self.scheme = scheme
        super().__init__(equation=equation, state=state)

    @property
    def transporting_velocity(self):
        return "prognostic"

    def setup_fields(self):
        self.x = TimeLevelFields(self.equation, self.scheme.nlevels)

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


class SemiImplicitQuasiNewton(BaseTimestepper):
    """
    Implements a semi-implicit quasi-Newton discretisation,
    with Strang splitting and auxilliary semi-Lagrangian transport.
    """

    def __init__(self, equation_set, state, transport_schemes,
                 auxiliary_equations_and_schemes=None,
                 linear_solver=None,
                 diffusion_schemes=None,
                 physics_list=None, **kwargs):

        """
        Args:
            equation_set (:class:`PrognosticEquationSet`): the prognostic
                equation set to be solved
            state (:class:`State`) the model's state object
            transport_schemes: iterable of ``(field_name, scheme)`` pairs
                indicating the name of the field (str) to transport, and the
                :class:`TimeDiscretisation` to use
            auxiliary_equations_and_schemes
            linear_solver: a :class:`.TimesteppingSolver` object
            diffusion_schemes: optional iterable of ``(field_name, scheme)``
                pairs indicating the fields to diffuse, and the
                :class:`~.Diffusion` to use.
            physics_list: optional list of classes that implement `physics`
                schemes

        :kwargs: maxk is the number of outer iterations, maxi is the number
            of inner iterations and alpha is the offcentering parameter
    """

        self.maxk = kwargs.pop("maxk", 4)
        self.maxi = kwargs.pop("maxi", 1)
        self.alpha = kwargs.pop("alpha", 0.5)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

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

        super().__init__(equation_set, state)

        if auxiliary_equations_and_schemes is not None:
            for eqn, scheme in auxiliary_equations_and_schemes:
                self.x.add_fields(eqn)
                scheme.setup(eqn, self.transporting_velocity)
            self.auxiliary_schemes = [
                (eqn.field_name, scheme)
                for eqn, scheme in auxiliary_equations_and_schemes]
        else:
            self.auxiliary_schemes = []

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

    def setup_scheme(self):
        """Sets up transport and diffusion schemes"""
        # TODO: apply_bcs should be False for advection but this means
        # tests with KGOs fail
        apply_bcs = True
        for _, scheme in self.active_transport:
            scheme.setup(self.equation, self.transporting_velocity, apply_bcs, transport)
        apply_bcs = True
        for _, scheme in self.diffusion_schemes:
            scheme.setup(self.equation, self.transporting_velocity, apply_bcs, diffusion)

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
                    self.linear_solver.solve(xrhs, dy)  # solves linear system and places result in state.dy

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

            if len(self.physics_list) > 0:
                for field in self.x.np1:
                    self.state.fields(field.name()).assign(field)

                for physics in self.physics_list:
                    physics.apply()

                # TODO hack to reproduce current behaviour - only
                # necessary if physics routines change field values in
                # state
                for field in self.x.np1:
                    field.assign(self.state.fields(field.name()))


class PrescribedTransport(Timestepper):
    """
    Implements a timeloop with a prescibed transporting velocity
    """
    def __init__(self, equation, scheme, state, physics_list=None,
                 prescribed_transporting_velocity=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation
            state (:class:`State`): the model's state object
            physics_list: optional list of classes that implement `physics`
                schemes
            prescribed_transporting_velocity: (optional) expression specifying
                the prescribed transporting velocity
        """

        super().__init__(equation, scheme, state)

        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

        if prescribed_transporting_velocity is not None:
            self.velocity_projection = Projector(
                prescribed_transporting_velocity(self.state.t),
                self.state.fields('u'))
        else:
            self.velocity_projection = None

    @property
    def transporting_velocity(self):
        return self.state.fields('u')

    def setup_fields(self):
        self.x = TimeLevelFields(self.equation, self.scheme.nlevels)

    def setup_scheme(self):
        self.scheme.setup(self.equation, self.transporting_velocity)

    def timestep(self):
        if self.velocity_projection is not None:
            self.velocity_projection.project()

        super().timestep()

        with timed_stage("Physics"):

            if len(self.physics_list) > 0:
                for field in self.x.np1:
                    self.state.fields(field.name()).assign(field)

                for physics in self.physics_list:
                    physics.apply()

                # TODO hack to reproduce current behaviour - only
                # necessary if physics routines change field values in
                # state
                for field in self.x.np1:
                    field.assign(self.state.fields(field.name()))
