from __future__ import absolute_import
from pyop2.profiling import timed_stage
from gusto.state import IncompressibleState, Expression
from firedrake import DirichletBC


class Timestepper(object):
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

        self.state = state
        self.advection_dict = advection_dict
        self.linear_solver = linear_solver
        self.forcing = forcing
        self.diffusion_dict = {}
        if diffusion_dict is not None:
            self.diffusion_dict.update(diffusion_dict)

        if(isinstance(self.state, IncompressibleState)):
            self.incompressible = True
        else:
            self.incompressible = False

    def _set_ubar(self):
        """
        Update ubar in the advection methods.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]

        for field, advection in self.advection_dict.iteritems():
            advection.ubar.assign(un + state.timestepping.alpha*(unp1-un))

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.state.xnp1.split()[0]

        dim = unp1.ufl_element().value_shape()[0]
        bc = ("0.0",)*dim
        M = unp1.function_space()
        bcs = [DirichletBC(M, Expression(bc), "bottom"),
               DirichletBC(M, Expression(bc), "top")]

        for bc in bcs:
            bc.apply(unp1)

    def run(self, t, tmax):
        state = self.state

        state.xn.assign(state.x_init)

        xstar_fields = {name: func for (name, func) in
                        zip(state.fieldlist, state.xstar.split())}
        xp_fields = {name: func for (name, func) in
                     zip(state.fieldlist, state.xp.split())}

        dt = state.timestepping.dt
        alpha = state.timestepping.alpha
        if state.mu is not None:
            mu_alpha = dt
        else:
            mu_alpha = None
        state.dump()

        while t < tmax + 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt
            with timed_stage("Apply forcing terms"):
                self.forcing.apply((1-alpha)*dt, state.xn, state.xn, state.xstar)
                state.xnp1.assign(state.xn)

            for k in range(state.timestepping.maxk):
                with timed_stage("Compute ubar"):
                    self._set_ubar()  # computes state.ubar from state.xn and state.xnp1

                with timed_stage("Advection"):
                    for field, advection in self.advection_dict.iteritems():
                        # advects a field from xstar and puts result in xp
                        advection.apply(xstar_fields[field], xp_fields[field])
                state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

                for i in range(state.timestepping.maxi):

                    with timed_stage("Apply forcing terms"):
                        self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                           state.xrhs, mu_alpha=mu_alpha,
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
                state.dump()

        state.diagnostic_dump()
