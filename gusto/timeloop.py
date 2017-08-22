from abc import ABCMeta, abstractmethod
from pyop2.profiling import timed_stage
from gusto.linear_solvers import IncompressibleSolver
from gusto.transport_equation import EulerPoincare
from gusto.advection import NoAdvection
from gusto.moving_mesh.utility_functions import spherical_logarithm
from firedrake import DirichletBC, Function, LinearVariationalProblem, LinearVariationalSolver


__all__ = ["Timestepper", "AdvectionTimestepper"]


class BaseTimestepper(object, metaclass=ABCMeta):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    """

    def __init__(self, state, advected_fields, mesh_generator=None):
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
        self.mesh_generator = mesh_generator
        self.dt = state.timestepping.dt

        # list of fields that are advected as part of the nonlinear iteration
        fieldlist = [name for name in self.advection_dict.keys() if name in state.fieldlist]

        if state.timestepping.move_mesh:
            self.X0 = Function(state.mesh.coordinates)
            self.X1 = Function(state.mesh.coordinates)
            self.Advection = MovingMeshAdvectionStep(
                state, fieldlist,
                state.xn, state.xnp1,
                advection_dict, state.timestepping.alpha,
                self.X0, self.X1)
        else:
            self.Advection = AdvectionStep(
                state, fieldlist,
                state.xn, state.xnp1,
                advection_dict, state.timestepping.alpha)

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
    def run(self):
        pass


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
                 diffused_fields=None, physics_list=None, mesh_generator=None):

        super(Timestepper, self).__init__(state, advected_fields, mesh_generator)
        self.linear_solver = linear_solver
        self.forcing = forcing
        if diffused_fields is None:
            self.diffused_fields = ()
        else:
            self.diffused_fields = tuple(diffused_fields)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

        if isinstance(self.linear_solver, IncompressibleSolver):
            self.incompressible = True
        else:
            self.incompressible = False

        state.xb.assign(state.xn)

    def run(self, t, tmax, pickup=False):
        state = self.state
        state.setup_diagnostics()

        xstar_fields = {name: func for (name, func) in
                        zip(state.fieldlist, state.xstar.split())}
        xp_fields = {name: func for (name, func) in
                     zip(state.fieldlist, state.xp.split())}
        # list of fields that are passively advected (and possibly diffused)
<<<<<<< HEAD
        passive_fieldlist = [name for name in self.advection_dict.keys() if name not in state.fieldlist]
||||||| merged common ancestors
        passive_fieldlist = [name for name in self.advection_dict.keys() if name not in state.fieldlist]
        # list of fields that are advected as part of the nonlinear iteration
        fieldlist = [name for name in self.advection_dict.keys() if name in state.fieldlist]
=======
        passive_advection = [(name, scheme) for name, scheme in self.advected_fields if name not in state.fieldlist]
        # list of fields that are advected as part of the nonlinear iteration
        active_advection = [(name, scheme) for name, scheme in self.advected_fields if name in state.fieldlist]
>>>>>>> master

        dt = self.dt
        alpha = state.timestepping.alpha

        if state.mu is not None:
            mu_alpha = [0., dt]
        else:
            mu_alpha = [None, None]

        with timed_stage("Dump output"):
            state.setup_dump(pickup)
            t = state.dump(t, pickup)

        while t < tmax - 0.5*dt:
            if state.output.Verbose:
                print("STEP", t, dt)

            if state.timestepping.move_mesh:
                # This is used as the "old mesh" domain in projections
                state.mesh_old.coordinates.dat.data[:] = self.X1.dat.data[:]
                self.X0.assign(self.X1)
                with timed_stage("Mesh generation"):
                    self.X1.assign(self.mesh_generator.get_new_mesh())

            t += dt

            state.t.assign(t)

            with timed_stage("Apply forcing terms"):
                self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                                   state.xstar, mu_alpha=mu_alpha[0])

            state.xnp1.assign(state.xn)

            for k in range(state.timestepping.maxk):
                # At the moment, this is automagically moving the mesh (if
                # appropriate), which is not ideal
                with timed_stage("Advection"):
                    self.Advection.apply(xstar_fields, xp_fields)

                state.xrhs.assign(0.)  # residual for the linear solve

                for i in range(state.timestepping.maxi):

                    with timed_stage("Apply forcing terms"):
                        self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                           state.xrhs, mu_alpha=mu_alpha[1],
                                           incompressible=self.incompressible)

                    state.xrhs -= state.xnp1

                    with timed_stage("Implicit solve"):
                        self.linear_solver.solve()  # solves linear system and places result in state.dy

                    state.xnp1 += state.dy

            self._apply_bcs()

            for name, advection in passive_advection:
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


class AdvectionTimestepper(BaseTimestepper):

    def __init__(self, state, advected_fields, physics_list=None, mesh_generator=None):
        super(AdvectionTimestepper, self).__init__(state, advected_fields, mesh_generator)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

    def run(self, t, tmax, x_end=None):
        state = self.state
        state.setup_diagnostics()

        dt = self.dt
        xn_fields = {name: func for (name, func) in
                     zip(state.fieldlist, state.xn.split())}
        xnp1_fields = {name: func for (name, func) in
                       zip(state.fieldlist, state.xnp1.split())}

        with timed_stage("Dump output"):
            state.setup_dump()
            state.dump(t)

        while t < tmax - 0.5*dt:
            if state.output.Verbose:
                print("STEP", t, dt)

            # Horrible hacky magic: for time dependent stuff, e.g.
            # expressions for u, or forcing, stick a ufl Constant onto
            # state.t_const, and we automatically update that inside
            # the time loop

            if hasattr(state, "t_const"):
                state.t_const.assign(t + 0.5*dt)

            t += dt

            # since advection velocity ubar is calculated from xn and xnp1
            state.xnp1.assign(state.xn)

            if state.timestepping.move_mesh:
                # This is used as the "old mesh" domain in projections
                state.mesh_old.coordinates.dat.data[:] = self.X1.dat.data[:]
                self.X0.assign(self.X1)
                with timed_stage("Mesh generation"):
                    self.X1.assign(self.mesh_generator.get_new_mesh())

            # At the moment, this is automagically moving the mesh (if
            # appropriate), which is not ideal
            with timed_stage("Advection"):
                self.Advection.apply(xn_fields, xnp1_fields)

            # technically the above advection could be done "in place",
            # but writing into a new field and assigning back is more
            # robust to future changes
            state.xn.assign(state.xnp1)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            with timed_stage("Dump output"):
                state.dump(t)

            with timed_stage("Dump output"):
                state.dump(t)

        if x_end is not None:
            return {field: getattr(state.fields, field) for field in x_end}


class AdvectionStep(object):
    def __init__(self, state, fieldlist, xn, xnp1, advected_fields, alpha):
        self.state = state
        self.fieldlist = fieldlist
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
    def __init__(self, state, fieldlist, xn, xnp1,
                 advected_fields, alpha, X0, X1):
        super(MovingMeshAdvectionStep, self).__init__(
            state, fieldlist, xn, xnp1, advected_fields, alpha)

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
                    LHS = eqn.mass_term(eqn.trial)
                    RHS = eqn.mass_term(x_in[field], domain=self.state.mesh_old)
                    prob = LinearVariationalProblem(LHS, RHS, x_in[field], constant_jacobian=False)
                    self._projections[field] = LinearVariationalSolver(prob)
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

            # put mesh_new into mesh so it gets into LHS of projections
            self.state.mesh.coordinates.assign(X1)

            if field in self.projections(self.x_mid).keys():
                self.projections(self.x_mid)[field].solve()

            # advect field on new mesh
            advection.update_ubar(self.alpha*(unp1 - self.v1_V1))
            advection.apply(self.x_mid[field], x_out[field])
