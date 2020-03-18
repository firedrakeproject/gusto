from abc import ABCMeta, abstractmethod
from firedrake import Function
from pyop2.profiling import timed_stage
from gusto.configuration import logger
from gusto.linear_solvers import IncompressibleSolver
from gusto.state import FieldCreator

__all__ = ["TimeLevelFields", "CrankNicolson", "PrescribedAdvection"]


class TimeLevelFields(object):

    def __init__(self, state, eqn_schemes, time_levels=None):
        default_time_levels = ("nm1", "n", "np1")
        if time_levels is not None:
            time_levels = tuple(time_levels) + default_time_levels
        else:
            time_levels = default_time_levels

        for level in time_levels:
            field_fs = []
            for eqn, _ in eqn_schemes:
                field_fs.append(tuple((eqn.field_name, eqn.function_space)))

            setattr(self, level, FieldCreator(field_fs))

    def initialise(self, state):
        for field in self.n:
            field.assign(state.fields(field.name()))
            self.np1(field.name()).assign(field)

    def update(self):
        for field in self.n:
            self.nm1(field.name()).assign(field)
            field.assign(self.np1(field.name()))


class BaseTimestepper(object, metaclass=ABCMeta):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advection_schemes: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffusion_schemes: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """

    def __init__(self, state, eqn_schemes=None, physics_list=None):

        self.state = state
        if eqn_schemes is None:
            self.eqn_schemes = ()
        else:
            self.eqn_schemes = tuple(eqn_schemes)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.x.np1.X.split()[0]

        bcs = self.state.bcs

        for bc in bcs:
            bc.apply(unp1)

    def setup_timeloop(self):
        self.x = TimeLevelFields(self.state, self.eqn_schemes)
        for eqn, scheme in self.eqn_schemes:
            scheme._setup(eqn, self.advecting_velocity)

    @abstractmethod
    def timestep(self):
        """
        Implement the timestep
        """
        xn = self.x.n
        xnp1 = self.x.np1

        for eqn, scheme in self.eqn_schemes:
            name = eqn.field_name
            scheme.apply(xn(name), xnp1(name))

    def run(self, t, tmax, pickup=False):
        """
        This is the timeloop. After completing the semi implicit step
        any passively advected fields are updated, implicit diffusion and
        physics updates are applied (if required).
        """

        state = self.state

        if pickup:
            t = state.pickup_from_checkpoint()

        state.setup_diagnostics()

        with timed_stage("Dump output"):
            state.setup_dump(t, tmax, pickup)

        dt = state.dt

        self.setup_timeloop()

        self.x.initialise(state)

        while t < tmax - 0.5*dt:
            logger.info("at start of timestep, t=%s, dt=%s" % (t, dt))

            t += dt
            state.t.assign(t)

            self.x.update()

            self.timestep()

            for field in self.x.np1:
                state.fields(field.name()).assign(field)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            with timed_stage("Dump output"):
                state.dump(t)

        if state.output.checkpoint:
            state.chkpt.close()

        logger.info("TIMELOOP complete. t=%s, tmax=%s" % (t, tmax))


class CrankNicolson(BaseTimestepper):
    """
    This class implements a Crank-Nicolson discretisation, with Strang
    splitting and auxilliary semi-Lagrangian advection.

    :arg state: a :class:`.State` object
    :arg advection_schemes: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
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

    def __init__(self, state, advection_schemes, linear_solver, forcing,
                 diffusion_schemes=None, physics_list=None,
                 prescribed_fields=None, **kwargs):

        self.maxk = kwargs.pop("maxk", 4)
        self.maxi = kwargs.pop("maxi", 1)
        self.alpha = kwargs.pop("alpha", 0.5)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        super().__init__(state, advection_schemes, diffusion_schemes, physics_list, prescribed_fields)

        self.xrhs = Function(state.W)
        self.dy = Function(state.W)
        self.linear_solver = linear_solver
        self.forcing = forcing

        if isinstance(self.linear_solver, IncompressibleSolver):
            self.incompressible = True
        else:
            self.incompressible = False

        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme) for name, scheme in advection_schemes if name in state.fieldlist]

    @property
    def advecting_velocity(self):
        xn = self.x.n
        xnp1 = self.x.np1
        # computes ubar from un and unp1
        return xn('u') + self.alpha*(xnp1('u')-xn('u'))

    @property
    def passive_advection(self):
        """
        Advected fields that are not part of the semi implicit step are
        passively advected
        """
        return [(name, scheme) for name, scheme in
                self.advection_schemes if name not in self.state.fieldlist]

    def setup_timeloop(self, state):
        self.x = TimeLevelFields(state, time_levels=("star", "p"))

    def timestep(self):
        dt = self.state.dt
        alpha = self.alpha
        xn = self.x.n
        xnp1 = self.x.np1
        xstar = self.x.star
        xp = self.x.p
        xrhs = self.xrhs
        dy = self.dy

        with timed_stage("Apply forcing terms"):
            self.forcing.apply((1-alpha)*dt, xn, xn, xstar, implicit=False)

        for k in range(self.maxk):

            with timed_stage("Advection"):
                for name, advection in self.active_advection:
                    # advects a field from xstar and puts result in xp
                    advection.apply(xstar(name), xp(name))

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(alpha*dt, xp, xnp1, xrhs, implicit=True,
                                       incompressible=self.incompressible)

                xrhs -= xnp1.X

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve(xrhs, dy)  # solves linear system and places result in state.dy

                xnp1.X += dy

            self._apply_bcs()

        for name, advection in self.passive_advection:
            field = getattr(xn, name)
            # first computes ubar from un and unp1
            advection.update_ubar(xn('u') + alpha*(xnp1('u')-xn('u')))
            # advects a field from xn and puts result in xnp1
            advection.apply(field, field)

        self.x.update()

        with timed_stage("Diffusion"):
            for name, diffusion in self.diffusion_schemes:

                diffusion.apply(field, field)


class PrescribedAdvection(BaseTimestepper):

    def __init__(self, state, eqn_schemes, physics_list=None,
                 prescribed_advecting_velocity=None):

        self.prescribed_advecting_velocity = prescribed_advecting_velocity
        super().__init__(state, eqn_schemes=eqn_schemes,
                         physics_list=physics_list)

    @property
    def advecting_velocity(self):
        return self.state.fields('u')

    def timestep(self):
        if self.prescribed_advecting_velocity is not None:
            self.state.fields('u').project(self.prescribed_advecting_velocity)

        super().timestep()
