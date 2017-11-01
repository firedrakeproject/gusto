from abc import ABCMeta, abstractmethod, abstractproperty
from pyop2.profiling import timed_stage
from gusto.linear_solvers import IncompressibleSolver
from firedrake import DirichletBC, Function


__all__ = ["CrankNicolson", "ImplicitMidpoint", "AdvectionDiffusion"]


class BaseTimestepper(object, metaclass=ABCMeta):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """

    def __init__(self, state, advected_fields=None, diffused_fields=None,
                 physics_list=None):

        self.state = state

        # create function on the mixed function space to hold values
        # at n+1 time level
        self.xnp1 = Function(state.W)

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
        unp1 = self.xnp1.split()[0]

        if unp1.function_space().extruded:
            M = unp1.function_space()
            bcs = [DirichletBC(M, 0.0, "bottom"),
                   DirichletBC(M, 0.0, "top")]

            for bc in bcs:
                bc.apply(unp1)

    def setup_timeloop(self, t, tmax, pickup):
        """
        Setup the timeloop by setting up diagnostics, dumping the fields and
        picking up from a previous run, if required
        """
        self.state.setup_diagnostics()
        with timed_stage("Dump output"):
            self.state.setup_dump(tmax, pickup)
            t = self.state.dump(t, pickup)
        return t

    @abstractmethod
    def semi_implicit_step(self):
        """
        Implement the semi implicit step for the timestepping scheme.
        """
        pass

    def run(self, t, tmax, pickup=False, **kwargs):
        """
        This is the timeloop. After completing the semi implicit step
        any passively advected fields are updated, implicit diffusion and
        physics updates are applied (if required).
        """

        t = self.setup_timeloop(t, tmax, pickup, **kwargs)

        state = self.state
        dt = state.timestepping.dt

        while t < tmax - 0.5*dt:
            if state.output.Verbose:
                print("STEP", t, dt)

            t += dt
            state.t.assign(t)

            self.xnp1.assign(state.xn)

            self.semi_implicit_step()

            for name, advection in self.passive_advection:
                field = getattr(state.fields, name)
                # first computes ubar from state.xn and xnp1
                advection.update_ubar(state.xn, self.xnp1, state.timestepping.alpha)
                # advects a field from xn and puts result in xnp1
                advection.apply(field, field)

            state.xn.assign(self.xnp1)

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


class SemiImplicitTimestepper(BaseTimestepper):
    """
    This is a base class for semi-implicit timestepping schemes.

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """
    def __init__(self, state, advected_fields, linear_solver, forcing=None,
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

        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme) for name, scheme in advected_fields if name in state.fieldlist]

        # create functions on the mixed function space to hold values
        # of the residual and the update
        self.xrhs = Function(state.W)
        self.dy = Function(state.W)

    @property
    def passive_advection(self):
        """
        Advected fields that are not part of the semi implicit step are
        passively advected
        """
        return [(name, scheme) for name, scheme in
                self.advected_fields if name not in self.state.fieldlist]


class ImplicitMidpoint(SemiImplicitTimestepper):
    """
    This class implements an implicit midpoint timestepper.

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """

    def setup_timeloop(self, t, tmax, pickup, **kwargs):
        self.maxk = kwargs.get("maxk", 4)

        t = super().setup_timeloop(t, tmax, pickup)

        # set up dictionaries to access the xn and xrhs functions separately
        self.xn_fields = {name: func for (name, func) in
                          zip(self.state.fieldlist, self.state.xn.split())}
        self.xrhs_fields = {name: func for (name, func) in
                            zip(self.state.fieldlist, self.xrhs.split())}
        return t

    def semi_implicit_step(self):
        state = self.state

        # xrhs is the residual which goes in the linear solve
        self.xrhs.assign(0.)

        for k in range(self.maxk):

            with timed_stage("Advection"):
                for name, advection in self.active_advection:
                    # update xbar for evaluating forcing terms and
                    # advecting velocity, or just update ubar
                    if hasattr(advection, "forcing"):
                        advection.update_xbar(state.xn, self.xnp1, state.timestepping.alpha)
                    else:
                        advection.update_ubar(state.xn, self.xnp1, state.timestepping.alpha)
                    # advects a field from xn and puts result in xrhs
                    advection.apply(self.xn_fields[name], self.xrhs_fields[name])

            self.xrhs -= self.xnp1

            with timed_stage("Implicit solve"):
                # solves linear system and places result in dy
                self.linear_solver.solve(self.xrhs, self.dy)

            self.xnp1 += self.dy


class CrankNicolson(SemiImplicitTimestepper):
    """
    This class implements a Crank-Nicolson discretisation, with Strang
    splitting and auxilliary semi-Lagrangian advection.

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """

    def setup_timeloop(self, t, tmax, pickup, **kwargs):
        self.maxi = kwargs.get("maxi", 1)
        self.maxk = kwargs.get("maxk", 4)

        t = super().setup_timeloop(t, tmax, pickup)

        # create functions on the mixed function space to hold values
        # of fields after first forcing update and advection update
        self.xstar = Function(self.state.W)
        self.xp = Function(self.state.W)

        # create dictionaries to access functions separately
        self.xstar_fields = {name: func for (name, func) in
                             zip(self.state.fieldlist, self.xstar.split())}
        self.xp_fields = {name: func for (name, func) in
                          zip(self.state.fieldlist, self.xp.split())}
        return t

    def semi_implicit_step(self):
        state = self.state
        dt = state.timestepping.dt
        alpha = state.timestepping.alpha

        with timed_stage("Apply forcing terms"):
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                               self.xstar, mu_alpha=self.mu_alpha[0])

        for k in range(self.maxk):

            with timed_stage("Advection"):
                for name, advection in self.active_advection:
                    # first computes ubar from state.xn and xnp1
                    advection.update_ubar(state.xn, self.xnp1, alpha)
                    # advects a field from xstar and puts result in xp
                    advection.apply(self.xstar_fields[name], self.xp_fields[name])

            # xrhs is the residual which goes in the linear solve
            self.xrhs.assign(0.)

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(alpha*dt, self.xp, self.xnp1,
                                       self.xrhs, mu_alpha=self.mu_alpha[1],
                                       incompressible=self.incompressible)

                self.xrhs -= self.xnp1

                with timed_stage("Implicit solve"):
                    # solves linear system and places result in dy
                    self.linear_solver.solve(self.xrhs, self.dy)

                self.xnp1 += self.dy

            self._apply_bcs()


class AdvectionDiffusion(BaseTimestepper):
    """
    This class implements a timestepper for the advection-diffusion equations.
    No semi implicit step is required.
    """

    @property
    def passive_advection(self):
        """
        All advected fields are passively advected
        """
        if self.advected_fields is not None:
            return self.advected_fields
        else:
            return []

    def semi_implicit_step(self):
        pass
