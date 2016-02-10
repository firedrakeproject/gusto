from abc import ABCMeta, abstractmethod

class Forcing(object):
    """
    Base class for forcing terms for dcore.

    :arg state: x :class:`.State` object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        self.state = state
    
    @abstractmethod
    def apply(self, scale, x, x_out):
        """
        Function takes x as input, computes F(x) and returns 
        x_out = x + scale*F(x) 
        as output.

        :arg scale: parameter to scale the output by.
        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass

class CompressibleForcing(Forcing):
    """
    Forcing class for compressible Euler equations.
    """

    def __init__(self, state):
        self.state = state

        self._build_forcing_solver()


    def _build_forcing_solver(self):
        """
        Only put forcing terms into the u equation.
        """
        
        state = self.state 
        V2 = state.V2
        W = state.W

        self.x0 = Function(W) #copy x to here

        u0,rho0,theta0 = split(self.x0)
        
        F = TrialFunction(V2)
        w = TestFunction(V2)
        self.uF = Function(V2)

        Omega = state.Omega
        cp = state.cp
        
        n = FacetNormal(state.mesh)
        pi = exner(theta0, rho0, state)
        
        a = inner(w,F)*dx
        L = (
            -inner(w,cross(2*Omega,u0))*dx #Coriolis term
            -cp*div(theta0*w)*pi*dx #pressure gradient (volume integral)
            +cp*jump(w*theta0,n)*avg(pi)*dS_v #pressure gradient (surface integral)
            +div(w)*state.Phi*dx #gravity term (Phi is geopotential)
        )

        bcs = [DirichletBC(V2, 0.0, "bottom"),
               DirichletBC(V2, 0.0, "top")]
        
        u_forcing_problem = LinearVariationalProblem(
            a,L,self.uF, bcs=bcs
        ) 

        self.u_forcing_solver = LinearVariationalSolver(u_forcing_problem)
        
    def apply(self, scaling, x_in, x_out):

        self.x0.assign(x_in)

        self.u_forcing_solver.solve() #places forcing in self.uF
        self.uF.scale(scaling)
        
        uF, _, _ = x_out.split()

        x_out.assign(x_in)
        uF += self.uF
        
def exner(theta,rho,state):
    """
    Compute the exner function.
    """
    R_d = state.R_d
    p_0 = state.p_0
    kappa = state.kappa
       
    return (R_d/p_0)**(kappa/(1-kappa))*pow(rho*theta, kappa/(1-kappa))

def exner_rho(theta,rho,state):
    R_d = state.R_d
    p_0 = state.p_0
    kappa = state.kappa
    
    return (R_d/p_0)**(kappa/(1-kappa))*pow(rho*theta, kappa/(1-kappa)-1)*theta*kappa/(1-kappa)

def exner_theta(theta,rho,state):
    R_d = state.R_d
    p_0 = state.p_0
    kappa = state.kappa
       
    return (R_d/p_0)**(kappa/(1-kappa))*pow(rho*theta, kappa/(1-kappa)-1)*rho*kappa/(1-kappa)
