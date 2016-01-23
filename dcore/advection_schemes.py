from abc import ABCMeta, abstractmethod

class AdvectionScheme(object):
    """
    Base class for advection schemes for dcore.

    :arg state: :class:`.State` object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        self.state = state
    
    @abstractmethod
    def apply(self, x, x_out, u):
        """
        Function takes x as input, and returns 
        x_out
        as output, advecting with velocity u.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass

class DGAdvectionScheme(object):
    """
    Advection schemes for a DG function space (vector or scalar).

    :arg state: :class:`.State` object.
    :arg V: :class:`.FunctionSpace` object (must be DG) where advected quantity lives.
    :arg Vu: :class:`.FunctionSpace` object where advecting velocity lives.
    :arg Continuity: If True, solve continuity equation, otherwise solve the advection equation. (Default: False)
    """

    def __init__(self, state, V, Vu, Continuity = False):
        self.state = state

        self.V = V
        self.x0 = Function(V)
        self.x1 = Function(V)
        self.dx = Function(V)

        self.ubar = Function(Vu)

        
        x = TestFunction(V)
        y = TrialFunction(V)

        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar,n) + abs(dot(ubar,n)))
        dt = state.dt
        
        #             - dt*div(phi*ubar)*pbar)*dx 
        #            + dt*dot(jump(phi),(un('+')*pbar('+') - un('-')*pbar('-')))*dS_v 
        #           + dt*dot(jump(phi),(un('+')*pbar('+') - un('-')*pbar('-')))*dS_h

        a = inner(x,y)*dx
        L = (
            inner(y,self.x1)*dx
        )
