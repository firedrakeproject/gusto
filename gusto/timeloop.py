from abc import ABCMeta, abstractmethod, abstractproperty
from pyop2.profiling import timed_stage
from gusto.linear_solvers import IncompressibleSolver
from firedrake import DirichletBC


__all__ = ["Timestepper", "AdvectionDiffusion"]


class BaseTimestepper(object, metaclass=ABCMeta):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    """

    def __init__(self, state, advected_fields=None, diffused_fields=None,
                 physics_list=None):

        self.state = state
        if advected_fields is None:
            self.advected_fields = ()
        else:
            self.advected_fields = tuple(advected_fields)
        if diffused_fields is None:
            self.diffused_fields = ()
        else:
            self.diffused_fields = tuple(diffused_fields)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

    @abstractproperty
    def passive_advection(self):
        """list of fields that are passively advected (and possibly diffused)"""
        pass

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.state.xnp1.split()[0]

        if unp1.function_space().extruded:
            M = unp1.function_space()
            bcs = [DirichletBC(M, 0.0, "bottom"),
                   DirichletBC(M, 0.0, "top")]

            for bc in bcs:
                bc.apply(unp1)

    def setup_timeloop(self, t, tmax, pickup):
        self.state.setup_diagnostics()
        with timed_stage("Dump output"):
            self.state.setup_dump(tmax, pickup)
            t = self.state.dump(t, pickup)
        return t

    @abstractmethod
    def nonlinear_timestep(self):
        pass

    def run(self, t, tmax, pickup=False):

        t = self.setup_timeloop(t, tmax, pickup)

        state = self.state
        dt = state.timestepping.dt

        while t < tmax - 0.5*dt:
            if state.output.Verbose:
                print("STEP", t, dt)

            t += dt
            state.t.assign(t)

            state.xnp1.assign(state.xn)

            self.nonlinear_timestep()

            for name, advection in self.passive_advection:
                field = getattr(state.fields, name)
                # first computes ubar from state.xn and state.xnp1
                advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
                # advects a field from xn and puts result in xnp1
                advection.apply(field, field)

            state.xb.assign(state.xn)
            state.xn.assign(state.xnp1)

            with timed_stage("Diffusion"):
                for name, diffusion in self.diffused_fields:
                    field = getattr(state.fields, name)
                    diffusion.apply(field, field)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            with timed_stage("Dump output"):
                state.dump(t, pickup=False)

        print("TIMELOOP complete. t= " + str(t) + " tmax=" + str(tmax))


class Timestepper(BaseTimestepper):
    """
    Build a timestepper to implement an "auxiliary semi-Lagrangian" timestepping
    scheme for the dynamical core.

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    """

    def __init__(self, state, advected_fields, linear_solver, forcing,
                 diffused_fields=None, physics_list=None):

        super().__init__(state, advected_fields, diffused_fields, physics_list)
        self.linear_solver = linear_solver
        self.forcing = forcing

        if isinstance(self.linear_solver, IncompressibleSolver):
            self.incompressible = True
        else:
            self.incompressible = False

        if state.mu is not None:
            self.mu_alpha = [0., state.timestepping.dt]
        else:
            self.mu_alpha = [None, None]

        self.xstar_fields = {name: func for (name, func) in
                             zip(state.fieldlist, state.xstar.split())}
        self.xp_fields = {name: func for (name, func) in
                          zip(state.fieldlist, state.xp.split())}

        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme) for name, scheme in advected_fields if name in state.fieldlist]

        state.xb.assign(state.xn)

    @property
    def passive_advection(self):
        return [(name, scheme) for name, scheme in
                self.advected_fields if name not in self.state.fieldlist]

    def nonlinear_timestep(self):
        state = self.state
        dt = state.timestepping.dt
        alpha = state.timestepping.alpha

        with timed_stage("Apply forcing terms"):
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                               state.xstar, mu_alpha=self.mu_alpha[0])

        for k in range(state.timestepping.maxk):

            with timed_stage("Advection"):
                for name, advection in self.active_advection:
                    # first computes ubar from state.xn and state.xnp1
                    advection.update_ubar(state.xn, state.xnp1, alpha)
                    # advects a field from xstar and puts result in xp
                    advection.apply(self.xstar_fields[name], self.xp_fields[name])

            state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(state.timestepping.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                       state.xrhs, mu_alpha=self.mu_alpha[1],
                                       incompressible=self.incompressible)

                state.xrhs -= state.xnp1

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve()  # solves linear system and places result in state.dy

                state.xnp1 += state.dy

            self._apply_bcs()


class AdvectionDiffusion(BaseTimestepper):

    @property
    def passive_advection(self):
        if self.advected_fields is not None:
            return self.advected_fields
        else:
            return []

    def nonlinear_timestep(self):
        pass
