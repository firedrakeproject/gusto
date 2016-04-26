class Timestepper(object):
    """
    Build a timestepper to implement an "auxiliary semi-Lagrangian" timestepping
    scheme for the dynamical core.

    :arg state: a :class:`.State` object
    :arg advection_list a list of tuples (scheme, i), where i is an
        :class:`.AdvectionScheme` object, and i is the index indicating
        which component of the mixed function space to advect.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    """

    def __init__(self, state, advection_list, linear_solver, forcing):

        self.state = state
        self.advection_list = advection_list
        self.linear_solver = linear_solver
        self.forcing = forcing

    def _set_ubar(self):
        """
        Update ubar in the advection methods.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]

        for advection, index in self.advection_list:
            advection.ubar.assign(un + state.timestepping.alpha*unp1)

    def run(self, t, tmax):
        state = self.state

        state.xn.assign(state.x_init)

        xstar_fields = state.xstar.split()
        xp_fields = state.xp.split()

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
                for advection, index in self.advection_list:
                    # advects a field from xstar and puts result in xp
                    advection.apply(xstar_fields[index], xp_fields[index])
                for i in range(state.timestepping.maxi):
                    state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve
                    self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                       state.xrhs)
                    state.xrhs -= state.xnp1
                    self.linear_solver.solve()  # solves linear system and places result in state.dy
                    state.xnp1 += state.dy

            state.xn.assign(state.xnp1)
            state.dump()
        state.diagnostic_dump()
