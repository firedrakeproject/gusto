from firedrake import *

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


    def run(self):
        t = self.t0
        dt = self.dt
        tmax = self.tmax
        
        self.xn.assign(self.x_init)

        xstar_fields = self.xstar.split()
        xp_fields = self.xp.split()

        while(t<tmax - 0.5*dt):
            t += dt 
            self.apply_forcing(self.xn,self.xstar)
            self.xnp1.assign(self.xn)
            
            for(k in range(self.maxk)):
                self.set_ubar()  #computes self.ubar from self.xn and self.xnp1
                for advection, index in self.advection_list:
                    advection.apply(xstar_fields[index],xp_fields[index]) #advects a field from xstar and puts result in xp
                for(i in range(self.maxi)):
                    self.xrhs.assign(0.) #xrhs is the residual which goes in the linear solve
                    self.apply_forcing(self.xp,self.xrhs)
                    self.xrhs -= self.xnp1
                    self.linear_system.solve() # solves linear system and places result in self.dy
                    self.xnp1 += self.dy
            
            self.xn.assign(self.xnp1)
