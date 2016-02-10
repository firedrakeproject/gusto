from abc import ABCMeta, abstractmethod
from firedrake import Function, TestFunction, TrialFunction, \
    LinearVariationalProblem, LinearVariationalSolver, FacetNormal, \
    dx, dot, grad, jump, avg, dS_v, dS_h

class Advection(object):
    """
    Base class for advection schemes for dcore.

    :arg state: :class:`.State` object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        self.state = state
    
    @abstractmethod
    def apply(self, x, x_out):
        """
        Function takes x as input, computes F(x) and returns x_out
        as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass    

class NoAdvection(Advection):
    """
    An non-advection scheme that does nothing.
    """

    def __init__(self, state):
        self.state = state

        #create a ubar field even though we don't use it.
        self.ubar = Function(state.V2)

    def apply(self, x_in, x_out):

        x_out.assign(x_in)

class LinearAdvection_Vt(Advection):
    """
    An advection scheme that uses the linearised background state 
    in evaluation of the advection term for the Vt temperature space.

    :arg state: :class:`.State` object.
    :arg qbar: :class:`.Function` object. The reference function that we 
    are linearising around.
    :arg options: a PETSc options dictionary
    """

    def __init__(self, state, V, qbar, options = None):
        self.state = state
        self.ubar = Function(state.V2)

        p = TestFunction(state.Vt)
        q = TrialFunction(state.Vt)
        
        dq = Function(state.Vt)

        a = p*q*dx
        k = state.k #Upward pointing unit vector
        L = -p*dot(self.ubar,k)*dot(k,grad(qbar))*dx

        aProblem = LinearVariationalProblem(a,L,dq)
        if options == None:
            options = {'ksp_type':'cg',
                       'pc_type':'bjacobi',
                       'sub_pc_type':'ilu'}
            
        self.solver = LinearVariationalSolver(aProblem,
                                              solver_parameters = options)

    def apply(self, x_in, x_out):
        dt = self.state.dt
        self.solver.solve()
        x_out.assign(x_in + dt*dq)


class LinearAdvection_V3(Advection):
    """
    An advection scheme that uses the linearised background state 
    in evaluation of the advection term for the V3 DG space.

    :arg state: :class:`.State` object.
    :arg qbar: :class:`.Function` object. The reference function that we 
    are linearising around.
    :arg options: a PETSc options dictionary
    """

    def __init__(self, state, qbar, options = None):
        self.state = state
        self.ubar = Function(state.V2)

        p = TestFunction(state.V3)
        q = TrialFunction(state.V3)
        
        dq = Function(state.V3)

        n = FacetNormal(state.mesh)
        
        a = p*q*dx
        L = (dot(grad(p), self.ubar)*q*dx
             - jump(self.ubar*p, n)*avg(q)*(dS_v + dS_h))

        aProblem = LinearVariationalProblem(a,L,dq)
        if options == None:
            options = {'ksp_type':'cg',
                       'pc_type':'bjacobi',
                       'sub_pc_type':'ilu'}
            
        self.solver = LinearVariationalSolver(aProblem,
                                              solver_parameters = options)

    def apply(self, x_in, x_out):
        dt = self.state.dt
        self.solver.solve()
        x_out.assign(x_in + dt*dq)
