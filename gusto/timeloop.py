from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from pyop2.profiling import timed_stage
from gusto.linear_solvers import IncompressibleSolver
from gusto.transport_equation import EulerPoincare
from gusto.advection import NoAdvection
from gusto.moving_mesh.utility_functions import spherical_logarithm
from firedrake import DirichletBC, Expression, Function, LinearVariationalProblem, LinearVariationalSolver, SpatialCoordinate, Projector


class BaseTimestepper(object):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advection_dict a dictionary with entries fieldname: scheme, where
        fieldname is the name of the field to be advection and scheme is an
        :class:`.AdvectionScheme` object
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, advection_dict, mesh_generator=None):
        # TODO: decide on consistent way of doing this. Ideally just
        # use mesh_generator, but seems tricky in other places.
        if state.timestepping.move_mesh:
            assert mesh_generator is not None
        if mesh_generator is not None:
            assert state.timestepping.move_mesh

        self.state = state
        self.advection_dict = advection_dict
        self.mesh_generator = mesh_generator
        self.dt = state.timestepping.dt

        # list of fields that are advected as part of the nonlinear iteration
        fieldlist = [name for name in self.advection_dict.keys() if name in state.fieldlist]

        if state.timestepping.move_mesh:
            mesh = state.mesh
            self.X0 = Function(mesh.coordinates.function_space()).interpolate(SpatialCoordinate(mesh))
            self.X1 = Function(mesh.coordinates.function_space()).interpolate(SpatialCoordinate(mesh))
            self.Advection = MovingMeshAdvectionManager(
                fieldlist,
                state.xn, state.xnp1,
                advection_dict, state.timestepping.alpha, state, self.X0, self.X1)
        else:
            self.Advection = AdvectionManager(
                fieldlist,
                state.xn, state.xnp1,
                advection_dict, state.timestepping.alpha)

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.state.xnp1.split()[0]

        if unp1.function_space().extruded:
            dim = unp1.ufl_element().value_shape()[0]
            bc = ("0.0",)*dim
            M = unp1.function_space()
            bcs = [DirichletBC(M, Expression(bc), "bottom"),
                   DirichletBC(M, Expression(bc), "top")]

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
    :arg advection_dict a dictionary with entries fieldname: scheme, where
        fieldname is the name of the field to be advection and scheme is an
        :class:`.AdvectionScheme` object
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    """

    def __init__(self, state, advection_dict, linear_solver, forcing, diffusion_dict=None, physics_list=None, mesh_generator=None):

        super(Timestepper, self).__init__(state, advection_dict, mesh_generator)
        self.linear_solver = linear_solver
        self.forcing = forcing
        self.diffusion_dict = {}
        if diffusion_dict is not None:
            self.diffusion_dict.update(diffusion_dict)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

        if(isinstance(self.linear_solver, IncompressibleSolver)):
            self.incompressible = True
        else:
            self.incompressible = False

    def run(self, t, tmax, pickup=False):
        state = self.state

        xstar_fields = {name: func for (name, func) in
                        zip(state.fieldlist, state.xstar.split())}
        xp_fields = {name: func for (name, func) in
                     zip(state.fieldlist, state.xp.split())}
        # list of fields that are passively advected (and possibly diffused)
        passive_fieldlist = [name for name in self.advection_dict.keys() if name not in state.fieldlist]

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
                print "STEP", t, dt

            if state.timestepping.move_mesh:
                state.mesh_old.coordinates.dat.data[:] = self.X1.dat.data[:]
                self.X0.assign(self.X1)
                self.X1.assign(self.mesh_generator.get_new_mesh())

            t += dt

            with timed_stage("Apply forcing terms"):
                self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                                   state.xstar, mu_alpha=mu_alpha[0])
                state.xnp1.assign(state.xn)

            for k in range(state.timestepping.maxk):
                if state.timestepping.move_mesh:
                    state.mesh.coordinates.assign(self.X0)

                with timed_stage("Advection"):
                    self.Advection.apply(xstar_fields, xp_fields)

                state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

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

            for name in passive_fieldlist:
                field = getattr(state.fields, name)
                advection = self.advection_dict[name]
                # first computes ubar from state.xn and state.xnp1
                advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
                # advects a field from xn and puts result in xnp1
                advection.apply(field, field)

            state.xn.assign(state.xnp1)

            with timed_stage("Diffusion"):
                for name, diffusion in self.diffusion_dict.iteritems():
                    field = getattr(state.fields, name)
                    diffusion.apply(field, field)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            with timed_stage("Dump output"):
                state.dump(t, pickup=False)

        state.diagnostic_dump()
        print "TIMELOOP complete. t= " + str(t) + " tmax=" + str(tmax)


class AdvectionTimestepper(BaseTimestepper):

    def __init__(self, state, advection_dict, physics_list=None, mesh_generator=None):

        super(AdvectionTimestepper, self).__init__(state, advection_dict, mesh_generator)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

    def run(self, t, tmax, x_end=None):
        state = self.state

        dt = self.dt
        xn_fields = {name: func for (name, func) in
                     zip(state.fieldlist, state.xn.split())}
        state.setup_dump()
        state.dump()

        while t < tmax - 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt
            state.xnp1.assign(state.xn)

            if state.timestepping.move_mesh:
                self.X0.assign(self.X1)
                self.X1.assign(self.mesh_generator.get_new_mesh())

            self.Advection.apply(xn_fields, xn_fields)

            for physics in self.physics_list:
                physics.apply()

            state.dump()

        state.diagnostic_dump()

        if x_end is not None:
            return {field: getattr(state.fields, field) for field in x_end}


class AdvectionManager(object):
    def __init__(self, fieldlist, xn, xnp1, advection_dict, alpha):
        self.fieldlist = fieldlist
        self.xn = xn
        self.xnp1 = xnp1
        self.advection_dict = advection_dict
        self.alpha = alpha

    def apply(self, x_in, x_out):
        for field, advection in self.advection_dict.iteritems():
            # first computes ubar from xn and xnp1
            un = self.xn.split()[0]
            unp1 = self.xnp1.split()[0]
            advection.update_ubar(un + self.alpha*(unp1-un))
            # advects field
            advection.apply(x_in[field], x_out[field])


class MovingMeshAdvectionManager(AdvectionManager):
    def __init__(self, fieldlist, xn, xnp1,
                 advection_dict, alpha, state, X0, X1):
        super(MovingMeshAdvectionManager, self).__init__(
            fieldlist, xn, xnp1, advection_dict, alpha)

        self.state = state
        x1 = state.xstar.copy()
        self.x1 = {name: func for (name, func) in
                   zip(state.fieldlist, x1.split())}
        self.X0 = X0
        self.X1 = X1

        self.v = Function(state.mesh.coordinates.function_space())
        self.v_V1 = Function(state.spaces("HDiv"))
        self.v1 = Function(state.mesh.coordinates.function_space())
        self.v1_V1 = Function(state.spaces("HDiv"))

    def projections(self, x_in):
        if not hasattr(self, "_projections"):
            self._projections = {}
            for field, advection in self.advection_dict.iteritems():
                if isinstance(advection, NoAdvection):
                    self._projections[field] = Projector(self.state.uexpr, x_in[field], constant_jacobian=not self.state.timestepping.move_mesh).solver
                elif (hasattr(advection.equation, "continuity") and advection.equation.continuity) or isinstance(advection.equation, EulerPoincare):
                    eqn = advection.equation
                    LHS = eqn.mass_term(eqn.trial)
                    RHS = eqn.mass_term(x_in[field], domain=self.state.mesh_old)
                    prob = LinearVariationalProblem(LHS, RHS, x_in[field], constant_jacobian=False)
                    self._projections[field] = LinearVariationalSolver(prob)
        return self._projections

    def apply(self, x_in, x_out):
        dt = self.state.timestepping.dt
        v_V1 = self.v_V1
        v1_V1 = self.v1_V1
        X0 = self.X0
        X1 = self.X1

        # Compute v (mesh velocity) and v1 (mesh velocity)
        # if self.state.on_sphere:
        if False:
            spherical_logarithm(X0, X1, self.v)
            self.v /= dt
            spherical_logarithm(X1, X0, self.v1)
            self.v1 /= -dt
            v_V1.project(self.v)
            v1_V1.project(self.v1)
        else:
            self.v_V1.project((X1-X0)/dt)
            self.v1_V1.project(-(X0-X1)/dt)
        un = self.xn.split()[0]
        unp1 = self.xnp1.split()[0]

        for field, advection in self.advection_dict.iteritems():
            advection.update_ubar((1-self.alpha)*(un-v_V1))
            self.state.mesh.coordinates.assign(X0)
            # advects field
            advection.apply(x_in[field], self.x1[field])

            # put mesh_new into mesh so it gets into LHS of projections
            self.state.mesh.coordinates.assign(X1)

            if field in self.projections(self.x1).keys():
                self.projections(self.x1)[field].solve()

            advection.update_ubar(self.alpha*(unp1-v1_V1))
            advection.apply(self.x1[field], x_out[field])
