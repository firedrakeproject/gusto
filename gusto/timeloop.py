from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from pyop2.profiling import timed_stage
from gusto.linear_solvers import IncompressibleSolver
from firedrake import DirichletBC, Expression
from mesh_movement import spherical_logarithm

class BaseTimestepper(object):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advection_dict a dictionary with entries fieldname: scheme, where
        fieldname is the name of the field to be advection and scheme is an
        :class:`.AdvectionScheme` object
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, advection_dict):

        self.state = state
        self.advection_dict = advection_dict

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

    def __init__(self, state, advection_dict, linear_solver, forcing, diffusion_dict=None, physics_list=None):

        super(Timestepper, self).__init__(state, advection_dict)
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
        # list of fields that are advected as part of the nonlinear iteration
        fieldlist = [name for name in self.advection_dict.keys() if name in state.fieldlist]

        if self.timestepping.move_mesh:
            Advection = MovingMeshAdvectionManager(
                xn, xnp1, xstar_fields, xp_fields,
                advection_dict, timestepping)
        else:
            Advection = AdvectionManager(
                xn, xnp1, xstar_fields, xp_fields,
                advection_dict, timestepping)
        
        dt = state.timestepping.dt
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
                state.mesh_old.coordinates.assign(mesh_new.coordinates)
                something here to compute the new mesh
                
            t += dt
            with timed_stage("Apply forcing terms"):

            if state.timestepping.move_mesh:
                state.mesh.coordinates.assign(mesh_old.coordinates)
                
                self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                                   state.xstar, mu_alpha=mu_alpha[0])
                state.xnp1.assign(state.xn)

            for k in range(state.timestepping.maxk):

                with timed_stage("Advection"):
                    Advection.apply()

                state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

                if state.timestepping.move_mesh:
                    state.mesh.coordinates.assign(mesh_new.coordinates)

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

    def __init__(self, state, advection_dict, physics_list=None):

        super(AdvectionTimestepper, self).__init__(state, advection_dict)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

    def run(self, t, tmax, x_end=None):
        state = self.state

        dt = state.timestepping.dt
        state.xnp1.assign(state.xn)

        state.setup_dump()
        state.dump()

        while t < tmax - 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt

            for name, advection in self.advection_dict.iteritems():
                field = getattr(state.fields, name)
                # first computes ubar from state.xn and state.xnp1
                advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
                # advects field
                advection.apply(field, field)

            for physics in self.physics_list:
                physics.apply()

            state.dump()

        state.diagnostic_dump()

        if x_end is not None:
            return {field: getattr(state.fields, field) for field in x_end}

class AdvectionManager(object):
    def __init__(self, xn, xnp1, xstar_fields, xp_fields,
                 advection_dict, timestepping, state):
        self.xn = xn
        self.xnp1 = xnp1
        self.xstar_fields = xstar_fields
        self.xp_fields = xp_fields
        self.advection_dict = advection_dict
        self.timestepping = timestepping
        self.state = state

    def apply(self):
        for field in fieldlist:
            advection = self.advection_dict[field]
            # first computes ubar from state.xn and state.xnp1
            un = self.xn.split()[0]
            unp1 = self.xnp1.split()[0]
            alpha = self.timestepping.alpha
            advection.update_ubar(un + alpha*(unp1-un))
            # advects a field from xstar and puts result in xp
            advection.apply(self.xstar_fields[field], self.xp_fields[field])

class MovingMeshAdvectionManager(AdvectionManager):
    def __init__(self, xn, xnp1, xstar_fields, xp_fields,
                 advection_dict, timestepping, state, X0, X1, dt):
        super(MovingMeshAdvectionManager, self).__init__(
            self, xn, xnp1, xstar_fields, xp_fields,
            advection_dict, timestepping, state)

        self.dt = dt
        
        self.v = X0.copy().assign(0.)
        self.v_V1 = Function(state.VHDiv)
        
        Build the solvers for the remapping
        
        
    def apply(self):
        for field in fieldlist:
            advection = self.advection_dict[field]
            # first computes ubar from state.xn and state.xnp1
            un = self.xn.split()[0]
            unp1 = self.xnp1.split()[0]
            alpha = self.timestepping.alpha

            X0 = self.X0
            X1 = self.X1
            dt = self.dt
            v = self.v
            v_V1 = self.v_V1
            
            if self.state.on_sphere:
                spherical_logarithm(X0,X1,v)
                v /= dt
            else:
                self.v.assign((X1-X0)/dt)
            v_V1.project(v)
                
            advection.update_ubar((1-alpha)*(un-v_V1))
            # advects a field from xstar and puts result in xp
            self.state.mesh.coordinates.assign(
                self.state.mesh_old.coordinates)
            advection.apply(self.xstar_fields[field], self.xstar_fields[field])

            self.state.mesh.coordinates.assign(
                self.state.mesh_new.coordinates)

            if self.state.on_sphere:
                spherical_logarithm(X1,X0,v)
                v /= -dt
            else:
                self.v.assign((X0-X1)/dt)
            v_V1.project(v)

            advection.update_ubar(alpha*(unp1-v_V1))
            advection.apply(self.xstar_fields[field], self.xp_fields[field])
