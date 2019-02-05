from abc import ABCMeta, abstractmethod, abstractproperty
from pyop2.profiling import timed_stage
from gusto.configuration import logger
from gusto.linear_solvers import IncompressibleSolver
from gusto.transport_equation import EulerPoincare
from gusto.advection import NoAdvection
from gusto.moving_mesh.utility_functions import spherical_logarithm
from firedrake import DirichletBC, Function, assemble
from firedrake.petsc import PETSc

__all__ = ["CrankNicolson", "AdvectionDiffusion"]


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
                 physics_list=None, mesh_generator=None):

        # TODO: decide on consistent way of doing this. Ideally just
        # use mesh_generator, but seems tricky in other places.
        if state.timestepping.move_mesh:
            assert mesh_generator is not None
        if mesh_generator is not None:
            assert state.timestepping.move_mesh

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

        self.mesh_generator = mesh_generator
        self.dt = state.timestepping.dt

    @abstractproperty
    def passive_advection(self):
        """list of fields that are passively advected (and possibly diffused)"""
        pass

    @abstractproperty
    def active_advection(self):
        """list of fields that are advected during the semi-implicit step"""
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
        """
        Setup the timeloop by setting up diagnostics, dumping the fields and
        picking up from a previous run, if required
        """
        state = self.state
        state.setup_diagnostics()
        with timed_stage("Dump output"):
            state.setup_dump(tmax, pickup)
            t = state.dump(t, pickup)

        self.passive_fields = {}
        for name, _ in self.passive_advection:
            self.passive_fields[name] = getattr(state.fields, name)

        if state.timestepping.move_mesh:
            self.X0 = Function(state.mesh.coordinates)
            self.X1 = Function(state.mesh.coordinates)
            self.Advection = MovingMeshAdvectionStep(
                state,
                state.xn, state.xnp1,
                self.active_advection, state.timestepping.alpha,
                self.X0, self.X1)
            if len(self.passive_advection) > 0:
                self.PassiveAdvection = MovingMeshAdvectionStep(
                    state,
                    state.xn, state.xnp1,
                    self.passive_advection, state.timestepping.alpha,
                    self.X0, self.X1)
        else:
            self.Advection = AdvectionStep(
                state,
                state.xn, state.xnp1,
                self.active_advection, state.timestepping.alpha)
            if len(self.passive_advection) > 0:
                self.PassiveAdvection = AdvectionStep(
                    state,
                    state.xn, state.xnp1,
                    self.passive_advection, state.timestepping.alpha)

        return t

    @abstractmethod
    def semi_implicit_step(self):
        """
        Implement the semi implicit step for the timestepping scheme.
        """
        pass

    def run(self, t, tmax, pickup=False):
        """
        This is the timeloop. After completing the semi implicit step
        any passively advected fields are updated, implicit diffusion and
        physics updates are applied (if required).
        """

        t = self.setup_timeloop(t, tmax, pickup)

        state = self.state
        dt = state.timestepping.dt

        while t < tmax - 0.5*dt:
            logger.info("at start of timestep, t=%s, dt=%s" % (t, dt))

            if state.timestepping.move_mesh:

                self.X0.assign(self.X1)

                self.mesh_generator.pre_meshgen_callback()
                with timed_stage("Mesh generation"):
                    self.X1.assign(self.mesh_generator.get_new_mesh())

            # Horrible hacky magic: for time dependent stuff, e.g.
            # expressions for u, or forcing, stick a ufl Constant onto
            # state.t_const, and we automatically update that inside
            # the time loop

            if hasattr(state, "t_const"):
                state.t_const.assign(t + 0.5*dt)

            t += dt

            state.t.assign(t)

            state.xnp1.assign(state.xn)

            self.semi_implicit_step()

            with timed_stage("PassiveAdvection"):
                if hasattr(self, "PassiveAdvection"):
                    self.PassiveAdvection.apply(self.passive_fields,
                                                self.passive_fields)

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

        if state.output.checkpoint:
            state.chkpt.close()

        logger.info("TIMELOOP complete. t=%s, tmax=%s" % (t, tmax))


class CrankNicolson(BaseTimestepper):
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

    def __init__(self, state, advected_fields, linear_solver, forcing,
                 diffused_fields=None, physics_list=None, mesh_generator=None):

        super().__init__(state, advected_fields, diffused_fields,
                         physics_list, mesh_generator)
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

        state.xb.assign(state.xn)

    @property
    def active_advection(self):
        """
        Advected fields that are not part of the semi implicit step are
        passively advected
        """
        return [(name, scheme) for name, scheme in
                self.advected_fields if name in self.state.fieldlist]

    @property
    def passive_advection(self):
        """
        Advected fields that are not part of the semi implicit step are
        passively advected
        """
        return [(name, scheme) for name, scheme in
                self.advected_fields if name not in self.state.fieldlist]

    def semi_implicit_step(self):
        state = self.state
        dt = state.timestepping.dt
        alpha = state.timestepping.alpha

        with timed_stage("Apply forcing terms"):
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                               state.xstar, implicit=False)

        for k in range(state.timestepping.maxk):

            if state.timestepping.move_mesh:
                self.X0.assign(self.X1)

                self.mesh_generator.pre_meshgen_callback()
                with timed_stage("Mesh generation"):
                    self.X1.assign(self.mesh_generator.get_new_mesh())

            # At the moment, this is automagically moving the mesh (if
            # appropriate), which is not ideal
            with timed_stage("Advection"):
                self.Advection.apply(self.xstar_fields, self.xp_fields)

            if state.timestepping.move_mesh:
                self.mesh_generator.post_meshgen_callback()

            state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(state.timestepping.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                       state.xrhs, implicit=True,
                                       incompressible=self.incompressible)

                state.xrhs -= state.xnp1

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve()  # solves linear system and places result in state.dy

                state.xnp1 += state.dy

            self._apply_bcs()


class AdvectionDiffusion(BaseTimestepper):
    """
    This class implements a timestepper for the advection-diffusion equations.
    No semi implicit step is required.
    """

    @property
    def active_advection(self):
        """
        Advected fields that are not part of the semi implicit step are
        passively advected
        """
        return []

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


class AdvectionStep(object):
    def __init__(self, state, xn, xnp1, advected_fields, alpha):
        self.state = state
        self.xn = xn
        self.xnp1 = xnp1
        self.advected_fields = advected_fields
        self.alpha = alpha

    def apply(self, x_in, x_out):

        un = self.xn.split()[0]
        unp1 = self.xnp1.split()[0]

        # Horrible hacky magic: if you want the velocity to be set
        # analytically, e.g. for an advection-only problem, put the
        # corresponding UFL expression in state.uexpr, and we will
        # use it here, splatting whatever was in un and unp1.
        #
        # Otherwise nothing happens here.

        if hasattr(self.state, "uexpr"):
            un.project(self.state.uexpr)
            unp1.assign(un)

        # Update ubar for each advection object
        for field, advection in self.advected_fields:
            advection.update_ubar((1 - self.alpha)*un + self.alpha*unp1)

        # Advect fields
        for field, advection in self.advected_fields:
            advection.apply(x_in[field], x_out[field])


class MovingMeshAdvectionStep(AdvectionStep):
    def __init__(self, state, xn, xnp1,
                 advected_fields, alpha, X0, X1):
        super(MovingMeshAdvectionStep, self).__init__(
            state, xn, xnp1, advected_fields, alpha)

        x_mid = Function(state.xstar.function_space())
        self.x_mid = {name: func for (name, func) in
                      zip(state.fieldlist, x_mid.split())}
        self.X0 = X0
        self.X1 = X1

        self.v = Function(state.mesh.coordinates.function_space())
        self.v_V1 = Function(state.spaces("HDiv"))
        self.v1 = Function(state.mesh.coordinates.function_space())
        self.v1_V1 = Function(state.spaces("HDiv"))

    def projections(self, x_in):
        if not hasattr(self, "_projections"):
            self._projections = {}
            for field, advection in self.advected_fields:
                if isinstance(advection, NoAdvection):
                    pass
                elif (hasattr(advection.equation, "continuity") and advection.equation.continuity) or isinstance(advection.equation, EulerPoincare):
                    eqn = advection.equation
                    lhs = eqn.mass_term(eqn.trial)
                    rhs = eqn.mass_term(x_in[field])
                    self._projections[field] = (lhs, rhs)
        return self._projections

    def apply(self, x_in, x_out):
        dt = self.state.timestepping.dt
        X0 = self.X0
        X1 = self.X1

        # Compute v (mesh velocity w.r.t. initial mesh) and
        # v1 (mesh velocity w.r.t. final mesh)
        # TODO: use Projectors below!
        if self.state.on_sphere:
            spherical_logarithm(X0, X1, self.v, self.state.mesh._radius)
            self.v /= dt
            spherical_logarithm(X1, X0, self.v1, self.state.mesh._radius)
            self.v1 /= -dt

            self.state.mesh.coordinates.assign(X0)
            self.v_V1.project(self.v)

            self.state.mesh.coordinates.assign(X1)
            self.v1_V1.project(self.v1)

        else:
            self.state.mesh.coordinates.assign(X0)
            self.v_V1.project((X1 - X0)/dt)

            self.state.mesh.coordinates.assign(X1)
            self.v1_V1.project(-(X0 - X1)/dt)

        un = self.xn.split()[0]
        unp1 = self.xnp1.split()[0]

        # Horrible hacky magic: if you want the velocity to be set
        # analytically, e.g. for an advection-only problem, put the
        # corresponding UFL expression in state.uexpr, and we will
        # use it here, splatting whatever was in un and unp1.

        if hasattr(self.state, "uexpr"):
            self.state.mesh.coordinates.assign(X0)
            un.project(self.state.uexpr)
            self.state.mesh.coordinates.assign(X1)
            unp1.project(self.state.uexpr)

        for field, advection in self.advected_fields:
            # advect field on old mesh
            self.state.mesh.coordinates.assign(X0)
            advection.update_ubar((1 - self.alpha)*(un - self.v_V1))
            advection.apply(x_in[field], self.x_mid[field])

            if field in self.projections(self.x_mid).keys():
                lhs, rhs = self.projections(self.x_mid)[field]
                with assemble(rhs).dat.vec as v:
                    Lvec = v

                self.state.mesh.coordinates.assign(X1)

                amat = assemble(lhs)
                amat.force_evaluation()
                a = amat.M.handle
                ksp = PETSc.KSP().create()
                ksp.setOperators(a)
                ksp.setFromOptions()

                with self.x_mid[field].dat.vec as x:
                    ksp.solve(Lvec, x)

            self.state.mesh.coordinates.assign(X1)
            # advect field on new mesh
            advection.update_ubar(self.alpha*(unp1 - self.v1_V1))
            advection.apply(self.x_mid[field], x_out[field])
