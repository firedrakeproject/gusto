from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from pyop2.profiling import timed_stage
from gusto.state import IncompressibleState
from firedrake import DirichletBC, Expression, Function


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

    def __init__(self, state, advection_dict, linear_solver, forcing, diffusion_dict=None):

        super(Timestepper, self).__init__(state, advection_dict)
        self.linear_solver = linear_solver
        self.forcing = forcing
        self.diffusion_dict = {}
        if diffusion_dict is not None:
            self.diffusion_dict.update(diffusion_dict)

        if(isinstance(self.state, IncompressibleState)):
            self.incompressible = True
        else:
            self.incompressible = False

    def run(self, t, tmax, pickup=False):
        state = self.state

        state.xn.assign(state.x_init)

        xstar_fields = {name: func for (name, func) in
                        zip(state.fieldlist, state.xstar.split())}
        xp_fields = {name: func for (name, func) in
                     zip(state.fieldlist, state.xp.split())}

        dt = state.timestepping.dt
        alpha = state.timestepping.alpha
        if state.mu is not None:
            mu_alpha = [0., dt]
        else:
            mu_alpha = [None, None]

        with timed_stage("Dump output"):
            t = state.dump(t, pickup)

        while t < tmax + 0.5*dt:
            state.t = t
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt
            with timed_stage("Apply forcing terms"):
                self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                                   state.xstar, mu_alpha=mu_alpha[0])
                state.xnp1.assign(state.xn)

            for k in range(state.timestepping.maxk):

                with timed_stage("Advection"):
                    for field, advection in self.advection_dict.iteritems():
                        # first computes ubar from state.xn and state.xnp1
                        advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
                        # advects a field from xstar and puts result in xp
                        advection.apply(xstar_fields[field], xp_fields[field])
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
            state.xn.assign(state.xnp1)

            with timed_stage("Diffusion"):
                for name, diffusion in self.diffusion_dict.iteritems():
                    diffusion.apply(state.field_dict[name], state.field_dict[name])

            with timed_stage("Dump output"):
                state.dump(t, pickup=False)

        state.diagnostic_dump()
        print "TIMELOOP complete. t= "+str(t-dt)+" tmax="+str(tmax)


class AdvectionTimestepper(BaseTimestepper):

    def run(self, t, tmax, x_end=None):
        state = self.state

        state.xn.assign(state.x_init)

        xn_fields = state.field_dict
        xnp1_fields = {name: func for (name, func) in
                       zip(state.fieldlist, state.xnp1.split())}
        for name, func in state.field_dict.iteritems():
            if name not in state.fieldlist and name in self.advection_dict.keys():
                xnp1_fields[name] = Function(func.function_space())

        dt = state.timestepping.dt
        state.xnp1.assign(state.xn)
        for name, func in state.field_dict.iteritems():
            if name not in state.fieldlist:
                xnp1_fields[name].assign(xn_fields[name])

        state.dump()

        while t < tmax + 0.5*dt:
            state.t = t
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt

            for field, advection in self.advection_dict.iteritems():
                # first computes ubar from state.xn and state.xnp1
                advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
                # advects a field from xn and puts result in xnp1
                advection.apply(xn_fields[field], xnp1_fields[field])

            state.xn.assign(state.xnp1)
            for name, func in state.field_dict.iteritems():
                if name not in state.fieldlist and name in self.advection_dict.keys():
                    xn_fields[name].assign(xnp1_fields[name])

            state.dump(t, pickup=False)

        state.diagnostic_dump()

        if x_end is not None:
            return {field: state.field_dict[field] for field in x_end}
