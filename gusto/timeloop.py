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

    def __init__(self, equation, method, state):
        self.equation = equation
        self.state = state

        self.setup_timeloop()

        self.schemes = []
        if type(method) is tuple:
            for scheme, apply_bcs, *active_labels in method:
                scheme.setup(equation, self.transporting_velocity, apply_bcs, *active_labels)
                self.schemes.append((equation.field_name, scheme))
        else:
            scheme = method
            scheme.setup(equation, self.transporting_velocity)
            self.schemes.append((equation.field_name, scheme))

    @abstractproperty
    def transporting_velocity(self):
        return NotImplementedError

    def setup_timeloop(self):
        self.x = TimeLevelFields(self.equation)

    @abstractmethod
    def timestep(self):
        return NotImplementedError

    def run(self, t, tmax, pickup=False):
        """
        This is the timeloop. After completing the semi implicit step
        any passively transported fields are updated, implicit diffusion and
        physics updates are applied (if required).
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
    Basic timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg transport_schemes: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to transport, and the
        :class:`~.TimeDiscretisation` to use.
    :arg diffusion_schemes: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """

    @property
    def transporting_velocity(self):
        return "prognostic"

    def timestep(self):
        """
        Implement the timestep
        """
        xn = self.x.n
        xnp1 = self.x.np1

        for name, scheme in self.schemes:
            scheme.apply(xnp1(name), xn(name))
            xn(name).assign(xnp1(name))


class SemiImplicitQuasiNewton(BaseTimestepper):
    """
    This class implements a Crank-Nicolson discretisation, with Strang
    splitting and auxilliary semi-Lagrangian transport.

    :arg state: a :class:`.State` object
    :arg transport_schemes: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to transport, and the
        :class:`~.TimeDiscretisation` to use.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    :arg diffusion_schemes: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    :arg prescribed_fields: an order list of tuples, pairing a field name with a
         function that returns the field as a function of time.
    :kwargs: maxk is the number of outer iterations, maxi is the number of inner
             iterations and alpha is the offcentering parameter
    """

    def __init__(self, equation_set, state, transport_schemes,
                 auxiliary_equations_and_schemes=None,
                 linear_solver=None,
                 diffusion_schemes=None,
                 physics_list=None, **kwargs):

        self.maxk = kwargs.pop("maxk", 4)
        self.maxi = kwargs.pop("maxi", 1)
        self.alpha = kwargs.pop("alpha", 0.5)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        self.equation_set = equation_set

        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

        schemes = []
        self.transport_schemes = []
        self.active_transport = []
        for scheme in transport_schemes:
            apply_bcs = False
            schemes.append((scheme, apply_bcs, transport))
            assert scheme.field_name in equation_set.field_names
            self.active_transport.append((scheme.field_name, scheme))

        self.diffusion_schemes = []
        if diffusion_schemes is not None:
            for scheme in diffusion_schemes:
                apply_bcs = True
                assert scheme.field_name in equation_set.field_names
                schemes.append((scheme, apply_bcs, diffusion))
                self.diffusion_schemes.append((scheme.field_name, scheme))

        super().__init__(equation_set, tuple(schemes), state)

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
        xn = self.x.n
        xnp1 = self.x.np1
        # computes ubar from un and unp1
        return xn('u') + self.alpha*(xnp1('u')-xn('u'))

    def setup_timeloop(self):
        super().setup_timeloop()
        self.x.add_fields(self.equation_set, time_levels=("star", "p"))

    def copy_active_tracers(self, x_in, x_out):
        """
        Copies active tracers from one set of fields to another, if those fields
        are not included in the linear solver. This is determined by whether the
        time derivative term for that tracer has a linearisation.

        :arg x_in:  The input set of fields
        :arg x_out: The output set of fields
        """

        for name in self.tracers_to_copy:
            x_out(name).assign(x_in(name))

    def timestep(self):
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

    def __init__(self, equation, method, state, physics_list=None,
                 prescribed_transporting_velocity=None):

        super().__init__(equation, method, state)

        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

        if prescribed_transporting_velocity is not None:
            self.velocity_projection = Projector(prescribed_transporting_velocity(self.state.t),
                                                 self.state.fields('u'))
        else:
            self.velocity_projection = None

    @property
    def transporting_velocity(self):
        return self.state.fields('u')

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
