from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function


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

    def _set_ubar(self):
        """
        Update ubar in the advection methods.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]

        for field, advection in self.advection_dict.iteritems():
            advection.ubar.assign(un + state.timestepping.alpha*(unp1-un))

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

    def run(self, t, tmax):
        state = self.state

        state.xn.assign(state.x_init)

        xstar_fields = {name: func for (name, func) in
                        zip(state.fieldlist, state.xstar.split())}
        xp_fields = {name: func for (name, func) in
                     zip(state.fieldlist, state.xp.split())}

        dt = state.timestepping.dt
        alpha = state.timestepping.alpha
        state.dump()

        while t < tmax + 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn, state.xstar)
            state.xnp1.assign(state.xn)

            for k in range(state.timestepping.maxk):
                self._set_ubar()  # computes state.ubar from state.xn and state.xnp1
                for field, advection in self.advection_dict.iteritems():
                    # advects a field from xstar and puts result in xp
                    advection.apply(xstar_fields[field], xp_fields[field])
                state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve
                for i in range(state.timestepping.maxi):
                    self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                       state.xrhs)
                    state.xrhs -= state.xnp1
                    self.linear_solver.solve()  # solves linear system and places result in state.dy
                    state.xnp1 += state.dy

            state.xn.assign(state.xnp1)

            for name, diffusion in self.diffusion_dict.iteritems():
                diffusion.apply(state.field_dict[name], state.field_dict[name])

            state.dump()

        state.diagnostic_dump()


class MovingMeshTimestepper(BaseTimestepper):
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
    def __init__(self, state, advection_dict, moving_mesh_advection, linear_solver, forcing, diffusion_dict=None):

        super(MovingMeshTimestepper, self).__init__(state, advection_dict)
        self.moving_mesh_advection = moving_mesh_advection
        self.linear_solver = linear_solver
        self.forcing = forcing
        self.diffusion_dict = {}
        if diffusion_dict is not None:
            self.diffusion_dict.update(diffusion_dict)

    def run(self, t, tmax):
        state = self.state

        state.xn.assign(state.x_init)

        xstar_fields = {name: func for (name, func) in
                        zip(state.fieldlist, state.xstar.split())}
        xp_fields = {name: func for (name, func) in
                     zip(state.fieldlist, state.xp.split())}

        dt = state.timestepping.dt
        alpha = state.timestepping.alpha
        state.dump()

        while t < tmax + 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn, state.xstar)
            state.xnp1.assign(state.xn)

            for k in range(state.timestepping.maxk):

                self.moving_mesh_advection.advection(xstar_fields, xp_fields, t, k)
                state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve
                for i in range(state.timestepping.maxi):
                    self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                       state.xrhs)
                    state.xrhs -= state.xnp1
                    self.linear_solver.solve()  # solves linear system and places result in state.dy
                    state.xnp1 += state.dy

            state.xn.assign(state.xnp1)

            for name, diffusion in self.diffusion_dict.iteritems():
                diffusion.apply(state.field_dict[name], state.field_dict[name])

            state.dump()

        state.diagnostic_dump()


class AdvectionTimestepper(BaseTimestepper):

    def run(self, t, tmax, x_end=None):
        state = self.state

        state.xn.assign(state.x_init)

        xn_fields = state.field_dict
        xnp1_fields = {name: func for (name, func) in
                       zip(state.fieldlist, state.xnp1.split())}
        for name, func in state.field_dict.iteritems():
            if name not in state.fieldlist:
                xnp1_fields[name] = Function(func.function_space())

        dt = state.timestepping.dt
        state.xnp1.assign(state.xn)
        for name, func in state.field_dict.iteritems():
            if name not in state.fieldlist:
                xnp1_fields[name].assign(xn_fields[name])

        state.dump()

        while t < tmax + 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt

            self._set_ubar()  # computes state.ubar from state.xn and state.xnp1
            for field, advection in self.advection_dict.iteritems():
                # advects a field from xn and puts result in xnp1
                advection.apply(xn_fields[field], xnp1_fields[field])

            state.xn.assign(state.xnp1)
            for name, func in state.field_dict.iteritems():
                if name not in state.fieldlist:
                    xn_fields[name].assign(xnp1_fields[name])

            state.dump()

        state.diagnostic_dump()

        if x_end is not None:
            return {field: state.field_dict[field] for field in x_end}


class MovingMeshAdvectionTimestepper(BaseTimestepper):

    def __init__(self, state, advection_dict, moving_mesh_advection):
        super(MovingMeshAdvectionTimestepper, self).__init__(state, advection_dict)
        self.moving_mesh_advection = moving_mesh_advection

    def run(self, t, tmax, x_end=None):
        state = self.state

        state.xn.assign(state.x_init)

        xn_fields = state.field_dict
        xnp1_fields = {name: func for (name, func) in
                       zip(state.fieldlist, state.xnp1.split())}

        dt = state.timestepping.dt
        state.xnp1.assign(state.xn)
        state.dump()

        while t < tmax + 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt

            self.moving_mesh_advection.advection(xn_fields, xnp1_fields, t)

            state.xn.assign(state.xnp1)

            state.dump()

        state.diagnostic_dump()

        if x_end is not None:
            return {field: state.field_dict[field] for field in x_end}
