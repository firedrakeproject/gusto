from firedrake import FiniteElement, TensorProductElement, HDiv, \
    FunctionSpace, MixedFunctionSpace, interval, triangle, Function

class State(object):
    """
    Build a model state to keep the variables in, and specify parameters.

    :arg mesh: The :class:`ExtrudedMesh` to use.
    :arg vertical_degree: integer, the degree for spaces in the vertical 
    (specifies the degree for the pressure space, other spaces are inferred)
    defaults to 1.
    :arg horizontal_degree: integer, the degree for spaces in the horizontal 
    (specifies the degree for the pressure space, other spaces are inferred)
    defaults to 1.
    :arg family: string, specifies the velocity space family to use. 
    Options:
    "RT": The Raviart-Thomas family (default, recommended for quads)
    "BDM": The BDM family
    "BDFM": The BDFM family
    :arg dt: the timestep
    :arg g: the acceleration due to gravity
    """

    def __init__(self, mesh, vertical_degree = 1, horizontal_degree = 1,
                 family = "RT",
                 dt = 1.0,
                 g = 9.81, k = None):
        
        #The mesh
        self.mesh = mesh

        #parameters
        self.dt = dt
        self.g = g
        if(k != None):
            self.k = k
        
        #Build the spaces
        self._build_spaces(mesh, vertical_degree,
                          horizontal_degree, family)
        
        #Allocate state
        self._allocate_state()

    def initialise(self, u0, rho0, theta0):
        """
        Initialise state variables from expressions.
        :arg u0: :class:`.Function` object, initial u
        :arg rho0: :class:`.Function` object, initial rho
        :arg theta0: :class:`.Function` object, initial theta
        """

        u_init, rho_init, theta_init = self.x_init.split()
        u_init.project(u0)
        rho_init.project(rho0)
        theta_init.project(theta0)

    def set_reference_profiles(self, rho_ref, theta_ref):
        """
        Initialise reference profiles
        :arg rho_ref: :class:`.Function` object, initial rho
        :arg theta_ref: :class:`.Function` object, initial theta
        """

        self.rho_b = Function(self.V3)
        self.theta_b = Function(self.Vt)

        self.rho_b.project(rho_ref)
        self.theta_b.project(theta_ref)        

    def _build_spaces(self, mesh, vertical_degree, horizontal_degree, family):
        """
        Build:
        velocity space self.V2,
        pressure space self.V3,
        temperature space self.Vt,
        mixed function space self.W = (V2,V3,Vt)
        """

        #horizontal base spaces
        cell = mesh._base_mesh.ufl_cell()
        print cell, cell.cellname()
        if(cell.cellname() == 'triangle'):
            cell = triangle
        print cell, type(cell)
        S1 = FiniteElement(family, cell, 2)
        S2 = FiniteElement("DG", cell, 1)

        #vertical base spaces
        T0 = FiniteElement("CG", interval, vertical_degree)
        T1 = FiniteElement("DG", interval, vertical_degree-1)

        #build spaces V2, V3, Vt
        V2h_elt = HDiv(TensorProductElement(S1, T1))
        V2t_elt = TensorProductElement(S2, T0)
        V3_elt = TensorProductElement(S2, T1)
        V2v_elt = HDiv(V2t_elt)
        V2_elt = V2h_elt + V2v_elt

        self.Vt = FunctionSpace(mesh, V2t_elt)
        self.V2 = FunctionSpace(mesh, V2_elt)
        self.V3 = FunctionSpace(mesh, V3_elt)

        self.W = MixedFunctionSpace((self.V2, self.V3, self.Vt))

    def _allocate_state(self):
        """
        Construct Functions to store the state variables.
        """

        W = self.W
        self.xn = Function(W)
        self.x_init = Function(W)
        self.xstar = Function(W)
        self.xp = Function(W)
        self.xnp1 = Function(W)
        self.xrhs = Function(W)
        self.dy = Function(W)

