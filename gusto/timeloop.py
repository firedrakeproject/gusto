from abc import ABCMeta, abstractmethod
from pyop2.profiling import timed_stage
from gusto.linear_solvers import IncompressibleSolver
from firedrake import DirichletBC


__all__ = ["Timestepper", "AdvectionTimestepper"]


class BaseTimestepper(object, metaclass=ABCMeta):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    """

    def __init__(self, model):

        self.model = model
        self.state = self.model.state
        if hasattr(model.physical_domain, "sponge_layer"):
            self.mu_alpha = [0., self.timestepping.dt]
        else:
            self.mu_alpha = [None, None]
        if model.advected_fields is None:
            self.advected_fields = ()
        else:
            self.advected_fields = tuple(model.advected_fields)
        if model.diffused_fields is None:
            self.diffused_fields = ()
        else:
            self.diffused_fields = tuple(model.diffused_fields)
        if model.physics_list is not None:
            self.physics_list = model.physics_list
        else:
            self.physics_list = []

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

    @abstractmethod
    def setup_timeloop(self):
        pass

    @abstractmethod
    def timeloop(self):
        pass

    def run(self, t, tmax, pickup=False):
        self.setup_timeloop(t, pickup)
        self.timeloop(t, tmax)


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

    def __init__(self, model):

        super(Timestepper, self).__init__(model)
        self.linear_solver = model.linear_solver
        self.forcing = model.forcing

        if isinstance(self.linear_solver, IncompressibleSolver):
            self.incompressible = True
        else:
            self.incompressible = False

    def setup_timeloop(self, t, pickup):
        state = self.model.state
        state.setup_diagnostics(self.model)
        self.xstar_fields = {name: func for (name, func) in
                             zip(state.fieldlist, state.xstar.split())}
        self.xp_fields = {name: func for (name, func) in
                          zip(state.fieldlist, state.xp.split())}
        # list of fields that are passively advected (and possibly diffused)
        self.passive_advection = [(name, scheme) for name, scheme in self.advected_fields if name not in state.fieldlist]
        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme) for name, scheme in self.advected_fields if name in state.fieldlist]

        # first dump
        with timed_stage("Dump output"):
            state.output.setup_dump(state, pickup)
            t = state.output.dump(state, t, pickup)

        state.xb.assign(state.xn)

    def timeloop(self, t, tmax):
        state = self.state

        dt = self.model.timestepping.dt
        alpha = self.model.timestepping.alpha

        while t < tmax - 0.5*dt:
            if state.output.output_params.Verbose:
                print("STEP", t, dt)

            t += dt
            state.t.assign(t)

            with timed_stage("Apply forcing terms"):
                self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                                   state.xstar, mu_alpha=self.mu_alpha[0])

            state.xnp1.assign(state.xn)

            for k in range(self.model.timestepping.maxk):

                with timed_stage("Advection"):
                    for name, advection in self.active_advection:
                        # first computes ubar from state.xn and state.xnp1
                        advection.update_ubar(state.xn, state.xnp1, alpha)
                        # advects a field from xstar and puts result in xp
                        advection.apply(self.xstar_fields[name], self.xp_fields[name])

                state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

                for i in range(self.model.timestepping.maxi):

                    with timed_stage("Apply forcing terms"):
                        self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                           state.xrhs, mu_alpha=self.mu_alpha[1],
                                           incompressible=self.incompressible)

                    state.xrhs -= state.xnp1

                    with timed_stage("Implicit solve"):
                        self.linear_solver.solve()  # solves linear system and places result in state.dy

                    state.xnp1 += state.dy

            self._apply_bcs()

            for name, advection in self.passive_advection:
                field = getattr(state.fields, name)
                # first computes ubar from state.xn and state.xnp1
                advection.update_ubar(state.xn, state.xnp1, self.timestepping.alpha)
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
                state.output.dump(state, t)

        print("TIMELOOP complete. t= " + str(t) + " tmax=" + str(tmax))


class AdvectionTimestepper(BaseTimestepper):

    def __init__(self, model):

        super(AdvectionTimestepper, self).__init__(model)

    def setup_timeloop(self, t, pickup=False):
        self.state.setup_diagnostics(self.model)

        with timed_stage("Dump output"):
            self.state.output.setup_dump(self.state)
            self.state.output.dump(self.state, t)

    def timeloop(self, t, tmax):
        state = self.model.state
        dt = self.model.timestepping.dt

        while t < tmax - 0.5*dt:
            if state.output.Verbose:
                print("STEP", t, dt)

            t += dt

            un = state.fields("u")
            with timed_stage("Advection"):
                for name, advection in self.advected_fields:
                    field = getattr(state.fields, name)
                    advection.update_ubar(un, un, 0.5)
                    # advects field
                    advection.apply(field, field)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            with timed_stage("Dump output"):
                state.output.dump(state, t)

        if x_end is not None:
            return {field: getattr(state.fields, field) for field in x_end}
