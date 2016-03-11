from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
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
    :arg g: the acceleration due to gravity
    """
    __metaclass__ = ABCMeta

    def __init__(self, mesh, vertical_degree=1, horizontal_degree=1,
                 family="RT",
                 dt=1.0,
                 alpha=0.5,
                 maxk=2,
                 maxi=2,
                 g=9.81,
                 cp=1004.5,
                 R_d=287,
                 p_0=1000.0 * 100.0,
                 kappa=2.0/7.0,
                 k=None,
                 Omega=None,
                 Verbose=False,
                 dumpfreq=10,
                 dumplist=(True,True,True)):

        # The mesh
        self.mesh = mesh

        # parameters
        self.dt = dt
        self.maxk = maxk
        self.maxi = maxi
        self.alpha = alpha
        self.g = g
        self.cp = cp
        self.R_d = R_d
        self.p_0 = p_0
        self.kappa = kappa
        if k is not None:
            self.k = k
        if Omega is not None:
            self.Omega = Omega

        self.Verbose = Verbose
        self.dumpfreq = dumpfreq
        self.dumplist = dumplist

        # Build the spaces
        self._build_spaces(mesh, vertical_degree,
                           horizontal_degree, family)

        # Allocate state
        self._allocate_state()

        self.dumped = False

    def dump(self):
        """
        Dump output
        """

        xn = self.xn.split()
        fieldlist = self.fieldlist

        if not self.dumped:
            self.dumpcount = 0
            self.Files = [0,0,0]
            self.xout = [0,0,0]
            for i in range(len(self.dumplist)):
                if(self.dumplist[i]):
                    (self.xout)[i] = Function(self.V[i])
                    self.Files[i] = File(fieldlist[i]+'.pvd')
                    self.xout[i].assign(xn[i])
                    self.Files[i] << self.xout[i]
            self.dumped = True
        else:
            self.dumpcount += 1
            print self.dumpcount, self.dumpfreq, 'DUMP STATS'
            if(self.dumpcount == self.dumpfreq):
                self.dumpcount = 0
                for i in range(len(self.dumplist)):
                    if(self.dumplist[i]):
                        print i
                        print self.Files[i], self.xout[i]
                        self.xout[i].assign(xn[i])
                        self.Files[i] << self.xout[i]

    def initialise(self, initial_conditions):
        """
        Initialise state variables
        """

        for i in range(self.x_init.function_space().num_sub_spaces()):
            self.x_init.sub(i).project(initial_conditions[i])

    @abstractmethod
    def _build_spaces(self, mesh, vertical_degree, horizontal_degree, family):
        """
        Build function spaces:
        """
        pass

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


class Compressible3DState(State):

    def __init__(self, mesh, vertical_degree=1, horizontal_degree=1,
                 family="RT",
                 dt=1.0,
                 alpha=0.5,
                 maxk=2,
                 maxi=2,
                 g=9.81,
                 cp=1004.5,
                 R_d=287,
                 p_0=1000.0 * 100.0,
                 kappa=2.0/7.0,
                 k=None,
                 Omega=None,
                 Verbose=False,
                 dumpfreq=10,
                 dumplist=(True,True,True)):

        super(Compressible3DState, self).__init__(mesh,
                                                  vertical_degree,
                                                  horizontal_degree,
                                                  family,
                                                  dt,
                                                  alpha,
                                                  maxk,
                                                  maxi,
                                                  g,
                                                  cp,
                                                  R_d,
                                                  p_0,
                                                  kappa,
                                                  k,
                                                  Omega,
                                                  Verbose,
                                                  dumpfreq,
                                                  dumplist)

        # build the geopotential
        V = FunctionSpace(mesh, "CG", 1)
        self.Phi = Function(V).interpolate(Expression("pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)"))
        self.Phi *= g

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


class ShallowWaterState(State):

    def _build_spaces(self, mesh, vertical, degree, family):

        cell = mesh.ufl_cell().cellname()

        V1_elt = FiniteElement(family, cell, degree)

        self.V = [0,0]
        self.V[0] = FunctionSpace(mesh,V1_elt)
        self.V[1] = FunctionSpace(mesh,"DG",degree-1)

        self.W = MixedFunctionSpace((self.V[0], self.V[1]))
