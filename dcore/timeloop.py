class Timestepper(object):
    """
    Build a timestepper to implement an "auxiliary semi-Lagrangian" timestepping
    scheme for the dynamical core.
    
    :arg state: a :class:`.State` object
    :arg advection_list a list of tuples (scheme, i), where i is an :class:`.AdvectionScheme` object, and i is the index indicating which component of the mixed function space to advect.
    """

    def __init__(self, state, advection_list):
    
        self.state = state
        self.advection_list = advection_list


    def run(self, t, dt, tmax):
        state = self.state
        
        state.xn.assign(state.x_init)

        xstar_fields = state.xstar.split()
        xp_fields = state.xp.split()

        while(t<tmax - 0.5*dt):
            t += dt 
            self.apply_forcing((1-state.alpha)*dt, state.xn, state.xstar)
            state.xnp1.assign(state.xn)
            
            for(k in range(state.maxk)):
                self.set_ubar()  #computes state.ubar from state.xn and state.xnp1
                for advection, index in self.advection_list:
                    advection.apply(xstar_fields[index], xp_fields[index]) #advects a field from xstar and puts result in xp
                for(i in range(state.maxi)):
                    state.xrhs.assign(0.) #xrhs is the residual which goes in the linear solve
                    self.apply_forcing(state.alpha*dt, state.xp, state.xrhs)
                    state.xrhs -= state.xnp1
                    self.linear_system.solve() # solves linear system and places result in state.dy
                    state.xnp1 += state.dy
            
            state.xn.assign(state.xnp1)
