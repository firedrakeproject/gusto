from abc import ABCMeta, abstractmethod
from pyop2.profiling import timed_stage
from gusto.configuration import logger
from gusto.linear_solvers import IncompressibleSolver

__all__ = ["CrankNicolson", "AdvectionDiffusion", "Advection", "Diffusion"]


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
    :arg prescribed_fields: an order list of tuples, pairing a field name with a
         function that returns the field as a function of time.
    """

    def __init__(self, state, advection_schemes=None, diffusion_schemes=None,
                 physics_list=None, prescribed_fields=None):

        self.state = state
        if advection_schemes is None:
            self.advection_schemes = ()
        else:
            self.advection_schemes = tuple(advection_schemes)
        if diffusion_schemes is None:
            self.diffusion_schemes = ()
        else:
            self.diffusion_schemes = tuple(diffusion_schemes)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []
        if prescribed_fields is not None:
            self.prescribed_fields = prescribed_fields
        else:
            self.prescribed_fields = []

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.state.xnp1.split()[0]

        bcs = self.state.bcs

        for bc in bcs:
            bc.apply(unp1)

    def setup_timeloop(self, state, t, tmax, pickup):
        """
        Setup the timeloop by setting up diagnostics, dumping the fields and
        picking up from a previous run, if required
        """
        if pickup:
            t = state.pickup_from_checkpoint()

        state.setup_diagnostics()

        with timed_stage("Dump output"):
            state.setup_dump(t, tmax, pickup)
        return t

    @abstractmethod
    def timestep(self):
        """
        Implement the timestep
        """
        pass

    def run(self, t, tmax, pickup=False):
        """
        This is the timeloop. After completing the semi implicit step
        any passively advected fields are updated, implicit diffusion and
        physics updates are applied (if required).
        """

        state = self.state

        t = self.setup_timeloop(state, t, tmax, pickup)

        dt = state.dt

        while t < tmax - 0.5*dt:
            logger.info("at start of timestep, t=%s, dt=%s" % (t, dt))

            t += dt
            state.t.assign(t)

            state.xnp1.assign(state.xn)

            for name, evaluation in self.prescribed_fields:
                state.fields(name).project(evaluation(state.t))

            self.timestep()

            state.xb.assign(state.xn)
            state.xn.assign(state.xnp1)

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

        self.linear_solver = linear_solver
        self.forcing = forcing

        if isinstance(self.linear_solver, IncompressibleSolver):
            self.incompressible = True
        else:
            self.incompressible = False

        self.xstar_fields = {name: func for (name, func) in
                             zip(state.fieldlist, state.xstar.split())}
        self.xp_fields = {name: func for (name, func) in
                          zip(state.fieldlist, state.xp.split())}

        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme) for name, scheme in advection_schemes if name in state.fieldlist]

        state.xb.assign(state.xn)

    @property
    def passive_advection(self):
        """
        Advected fields that are not part of the semi implicit step are
        passively advected
        """
        return [(name, scheme) for name, scheme in
                self.advection_schemes if name not in self.state.fieldlist]

    def timestep(self):
        state = self.state
        dt = state.dt
        alpha = self.alpha

        with timed_stage("Apply forcing terms"):
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                               state.xstar, implicit=False)

        for k in range(self.maxk):

            with timed_stage("Advection"):
                for name, advection in self.active_advection:
                    # first computes ubar from state.xn and state.xnp1
                    un = state.xn.split()[0]
                    unp1 = state.xnp1.split()[0]
                    advection.update_ubar(un + alpha*(unp1-un))
                    # advects a field from xstar and puts result in xp
                    advection.apply(self.xstar_fields[name], self.xp_fields[name])

            state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                       state.xrhs, implicit=True,
                                       incompressible=self.incompressible)

                state.xrhs -= state.xnp1

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve()  # solves linear system and places result in state.dy

                state.xnp1 += state.dy

            self._apply_bcs()

        for name, advection in self.passive_advection:
            field = getattr(state.fields, name)
            # first computes ubar from state.xn and state.xnp1
            un = state.xn.split()[0]
            unp1 = state.xnp1.split()[0]
            advection.update_ubar(un + alpha*(unp1-un))
            # advects a field from xn and puts result in xnp1
            advection.apply(field, field)

        with timed_stage("Diffusion"):
            for name, diffusion in self.diffusion_schemes:
                field = getattr(state.fields, name)
                diffusion.apply(field, field)


class AdvectionDiffusion(BaseTimestepper):
    """
    This class implements a splitting method for the advection-diffusion
    equations.
    """

    def timestep(self):

        state = self.state

        for name, scheme in self.advection_schemes:
            field = getattr(state.fields, name)
            scheme.update_ubar(state.fields('u'))
            scheme.apply(field, field)

        with timed_stage("Diffusion"):
            for name, scheme in self.diffusion_schemes:
                field = getattr(state.fields, name)
                scheme.apply(field, field)


class Advection(AdvectionDiffusion):

    def __init__(self, state, advection_schemes, physics_list=None,
                 prescribed_fields=None):
        super().__init__(state, advection_schemes,
                         diffusion_schemes=None,
                         physics_list=physics_list,
                         prescribed_fields=prescribed_fields)


class Diffusion(AdvectionDiffusion):

    def __init__(self, state, diffusion_schemes, physics_list=None,
                 prescribed_fields=None):
        super().__init__(state, advection_schemes=None,
                         diffusion_schemes=diffusion_schemes,
                         physics_list=physics_list,
                         prescribed_fields=prescribed_fields)
