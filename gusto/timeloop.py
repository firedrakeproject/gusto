from firedrake import Function
from pyop2.profiling import timed_stage
from gusto.configuration import logger
from gusto.forcing import Forcing
from gusto.form_manipulation_labelling import advection, diffusion
from gusto.linear_solvers import LinearTimesteppingSolver
from gusto.state import FieldCreator

__all__ = ["TimeLevelFields", "Timestepper", "CrankNicolson", "PrescribedAdvection"]


class TimeLevelFields(object):

    def __init__(self, state, equations, time_levels=None):
        default_time_levels = ("nm1", "n", "np1")
        if time_levels is not None:
            time_levels = tuple(time_levels) + default_time_levels
        else:
            time_levels = default_time_levels

        for level in time_levels:
            setattr(self, level, FieldCreator(equations))

    def initialise(self, state):
        for field in self.n:
            field.assign(state.fields(field.name()))
            self.np1(field.name()).assign(field)

    def update(self):
        for field in self.n:
            self.nm1(field.name()).assign(field)
            field.assign(self.np1(field.name()))


class Timestepper(object):
    """
    Basic timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advection_schemes: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffusion_schemes: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """

    def __init__(self, state, problem, physics_list=None):

        self.state = state

        self.equations = []
        self.schemes = []
        self.equations = [eqn for (eqn, _) in problem]
        self.setup_timeloop()
        for eqn, method in problem:
            if type(method) is tuple:
                for scheme, *active_labels in method:
                    scheme.setup(eqn, self.advecting_velocity, *active_labels)
                    self.schemes.append((eqn.field_name, scheme))
            else:
                scheme = method
                scheme.setup(eqn, self.advecting_velocity)
                self.schemes.append((eqn.field_name, scheme))

        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

    def advecting_velocity(self):
        return None

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.x.np1("u")

        for bc in self.bcs:
            bc.apply(unp1)

    def setup_timeloop(self):
        self.x = TimeLevelFields(self.state, self.equations)

    def timestep(self):
        """
        Implement the timestep
        """
        xn = self.x.n
        xnp1 = self.x.np1

        for name, scheme in self.schemes:
            scheme.apply(xn(name), xnp1(name))
            xn(name).assign(xnp1(name))

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


class CrankNicolson(Timestepper):
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

    def __init__(self, state, equation_set, advection_schemes,
                 linear_solver=None,
                 diffusion_schemes=None,
                 physics_list=None, **kwargs):

        self.maxk = kwargs.pop("maxk", 4)
        self.maxi = kwargs.pop("maxi", 1)
        self.alpha = kwargs.pop("alpha", 0.5)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        schemes = []
        self.advection_schemes = []
        for scheme in advection_schemes:
            schemes.append((scheme, advection))
            self.advection_schemes.append((scheme.field_name, scheme))
        if diffusion_schemes is None:
            self.diffusion_schemes = []
        for scheme in self.diffusion_schemes:
            schemes.append((scheme, diffusion))
            self.diffusion_schemes.append((scheme.field_name, scheme))
        super().__init__(state, [(equation_set, tuple(schemes))], physics_list)

        self.field_name = equation_set.field_name
        W = equation_set.function_space
        self.xrhs = Function(W)
        self.dy = Function(W)
        if linear_solver is None:
            self.linear_solver = LinearTimesteppingSolver(equation_set, state.dt, self.alpha)
        else:
            self.linear_solver = linear_solver
        self.forcing = Forcing(equation_set, state.dt, self.alpha)
        self.bcs = equation_set.bcs

        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme)
                                 for name, scheme in self.advection_schemes
                                 if name in equation_set.field_names]

        # list of fields that are passively advected
        self.passive_advection = [(name, scheme)
                                  for name, scheme in self.advection_schemes
                                  if name not in equation_set.field_names]

    @property
    def advecting_velocity(self):
        xn = self.x.n
        xnp1 = self.x.np1
        # computes ubar from un and unp1
        return xn('u') + self.alpha*(xnp1('u')-xn('u'))

    def setup_timeloop(self):
        self.x = TimeLevelFields(self.state, self.equations,
                                 time_levels=("star", "p"))

    def timestep(self):
        xn = self.x.n
        xnp1 = self.x.np1
        xstar = self.x.star
        xp = self.x.p
        xrhs = self.xrhs
        dy = self.dy

        with timed_stage("Apply forcing terms"):
            self.forcing.apply(xn, xn, xstar(self.field_name), "explicit")

        for k in range(self.maxk):

            with timed_stage("Advection"):
                for name, scheme in self.active_advection:
                    # advects a field from xstar and puts result in xp
                    scheme.apply(xstar(name), xp(name))

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(xp, xnp1, xrhs, "implicit")

                xrhs -= xnp1(self.field_name)

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve(xrhs, dy)  # solves linear system and places result in state.dy

                xnp1X = xnp1(self.field_name)
                xnp1X += dy

            self._apply_bcs()

        for name, scheme in self.passive_advection:
            field = getattr(xn, name)
            # advects a field from xn and puts result in xnp1
            scheme.apply(field, field)

        self.x.update()

        with timed_stage("Diffusion"):
            for name, scheme in self.diffusion_schemes:
                scheme.apply(field, field)


class PrescribedAdvection(Timestepper):

    def __init__(self, state, problem, physics_list=None,
                 prescribed_advecting_velocity=None):

        self.prescribed_advecting_velocity = prescribed_advecting_velocity
        super().__init__(state, problem,
                         physics_list=physics_list)

    @property
    def advecting_velocity(self):
        return self.state.fields('u')

    def timestep(self):
        if self.prescribed_advecting_velocity is not None:
            self.state.fields('u').project(self.prescribed_advecting_velocity)

        super().timestep()
