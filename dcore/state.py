from __future__ import absolute_import
from firedrake import FiniteElement, TensorProductElement, HDiv, \
    FunctionSpace, MixedFunctionSpace, interval, triangle, Function, \
    Expression, File


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
    """

    def __init__(self, mesh, vertical_degree=1, horizontal_degree=1,
                 family="RT",
                 timestepping=None,
                 output=None,
                 parameters=None):

        self.timestepping = timestepping
        self.output = output
        self.parameters = parameters

        # The mesh
        self.mesh = mesh

        # Build the spaces
        self._build_spaces(mesh, vertical_degree,
                           horizontal_degree, family)

        # build the geopotential
        V = FunctionSpace(mesh, "CG", 1)
        self.Phi = Function(V).interpolate(Expression("pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)"))
        self.Phi *= parameters.g

        # Allocate state
        self._allocate_state()

        self.dumped = False

    def dump(self):
        """
        Dump output
        """

        xn = self.xn.split()
        fieldlist = ('u','rho','theta')

        if not self.dumped:
            self.dumpcount = 0
            self.Files = [0,0,0]
            self.xout = [0,0,0]
            for i, dump in enumerate(self.output.dumplist):
                if(dump):
                    (self.xout)[i] = Function(self.V[i])
                    self.Files[i] = File(fieldlist[i]+'.pvd')
                    self.xout[i].assign(xn[i])
                    self.Files[i] << self.xout[i]
            self.dumped = True
        else:
            self.dumpcount += 1
            print self.dumpcount, self.output.dumpfreq, 'DUMP STATS'
            if(self.dumpcount == self.output.dumpfreq):
                self.dumpcount = 0
                for i, dump in enumerate(self.output.dumplist):
                    if(dump):
                        print i
                        print self.Files[i], self.xout[i]
                        self.xout[i].assign(xn[i])
                        self.Files[i] << self.xout[i]

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

        self.rhobar = Function(self.V[1])
        self.thetabar = Function(self.V[2])

        self.rhobar.project(rho_ref)
        self.thetabar.project(theta_ref)

    def _build_spaces(self, mesh, vertical_degree, horizontal_degree, family):
        """
        Build:
        velocity space self.V2,
        pressure space self.V3,
        temperature space self.Vt,
        mixed function space self.W = (V2,V3,Vt)
        """

        # horizontal base spaces
        cell = mesh._base_mesh.ufl_cell()
        if(cell.cellname() == 'triangle'):
            cell = triangle
        S1 = FiniteElement(family, cell, 2)
        S2 = FiniteElement("DG", cell, 1)

        # vertical base spaces
        T0 = FiniteElement("CG", interval, vertical_degree)
        T1 = FiniteElement("DG", interval, vertical_degree-1)

        # build spaces V2, V3, Vt
        V2h_elt = HDiv(TensorProductElement(S1, T1))
        V2t_elt = TensorProductElement(S2, T0)
        V3_elt = TensorProductElement(S2, T1)
        V2v_elt = HDiv(V2t_elt)
        V2_elt = V2h_elt + V2v_elt

        self.V = [0,0,0]
        self.V[0] = FunctionSpace(mesh, V2_elt)
        self.V[1] = FunctionSpace(mesh, V3_elt)
        self.V[2] = FunctionSpace(mesh, V2t_elt)

        self.W = MixedFunctionSpace((self.V[0], self.V[1], self.V[2]))

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
