from __future__ import absolute_import
from os import path
import itertools
from sys import exit
from abc import ABCMeta, abstractmethod
from firedrake import FiniteElement, TensorProductElement, HDiv, \
    FunctionSpace, MixedFunctionSpace, interval, triangle, Function, \
    Expression, File, TestFunction, TrialFunction, inner, div, FacetNormal, \
    ds_tb, dx, solve


class State(object):
    """
    Build a model state to keep the variables in, and specify parameters.

    :arg mesh: The :class:`Mesh` to use.
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
    :arg timestepping: class containing timestepping parameters
    :arg output: class containing output parameters
    :arg parameters: class containing physical parameters
    :arg fieldlist: list of prognostic field names

    """
    __metaclass__ = ABCMeta

    def __init__(self, mesh, vertical_degree=1, horizontal_degree=1,
                 family="RT",
                 timestepping=None,
                 output=None,
                 parameters=None,
                 fieldlist=None):

        self.timestepping = timestepping
        self.output = output
        self.parameters = parameters
        if fieldlist is None:
            raise RuntimeError("You must provide a fieldlist containing the names of the prognostic fields")
        else:
            self.fieldlist = fieldlist

        # The mesh
        self.mesh = mesh

        # Build the spaces
        self._build_spaces(mesh, vertical_degree,
                           horizontal_degree, family)

        if self.parameters.k is None:
            # build the vertical normal
            w = TestFunction(self.Vv)
            u = TrialFunction(self.Vv)
            self.parameters.k = Function(self.Vv)
            n = FacetNormal(self.mesh)
            krhs = -div(w)*self.parameters.z*dx + inner(w,n)*self.parameters.z*ds_tb
            klhs = inner(w,u)*dx
            solve(klhs == krhs, self.parameters.k)

        # Allocate state
        self._allocate_state()
        self.field_dict = {name: func for (name, func) in
                           zip(self.fieldlist, self.xn.split())}

        self.dumpfile = None

    def dump(self):
        """
        Dump output
        """

        # default behaviour is to dump all prognostic fields
        if self.output.dumplist is None:
            self.output.dumplist = self.fieldlist

        funcs = self.xn.split()
        to_dump = []
        for name, f in zip(self.fieldlist, funcs):
            if name in self.output.dumplist:
                to_dump.append(f)
            f.rename(name=name)

        dumpdir = path.join("results", self.output.dirname)

        outfile = path.join(dumpdir, "field_output.pvd")
        if self.dumpfile is None:
            if path.exists(dumpdir):
                exit("results directory '%s' already exists" % dumpdir)
            self.dumpcount = itertools.count()
            self.dumpfile = File(outfile, project_output=self.output.project_fields)

        if (next(self.dumpcount) % self.output.dumpfreq) == 0:
            self.dumpfile.write(*to_dump)

    def initialise(self, initial_conditions):
        """
        Initialise state variables
        """

        for x, ic in zip(self.x_init.split(), initial_conditions):
            x.project(ic)

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
                 timestepping=None,
                 output=None,
                 parameters=None,
                 fieldlist=None):

        super(Compressible3DState, self).__init__(mesh,
                                                  vertical_degree,
                                                  horizontal_degree,
                                                  family,
                                                  timestepping,
                                                  output,
                                                  parameters,
                                                  fieldlist)

        # build the geopotential
        V = FunctionSpace(mesh, "CG", 1)
        self.Phi = Function(V).interpolate(Expression("pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)"))
        self.Phi *= parameters.g

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
        if(cell.cellname() == 'interval'):
            cell = interval
        S1 = FiniteElement(family, cell, horizontal_degree+1)
        S2 = FiniteElement("DG", cell, horizontal_degree)

        # vertical base spaces
        T0 = FiniteElement("CG", interval, vertical_degree+1)
        T1 = FiniteElement("DG", interval, vertical_degree)

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

        self.Vv = FunctionSpace(mesh, V2v_elt)

        self.W = MixedFunctionSpace((self.V[0], self.V[1], self.V[2]))


class ShallowWaterState(State):

    def _build_spaces(self, mesh, vertical_degree, horizontal_degree, family):

        if vertical_degree is not None:
            raise ValueError('Mesh is not extruded in the vertical for shallow water')

        cell = mesh.ufl_cell().cellname()

        V1_elt = FiniteElement(family, cell, horizontal_degree)

        self.V = [0,0]
        self.V[0] = FunctionSpace(mesh,V1_elt)
        self.V[1] = FunctionSpace(mesh,"DG",horizontal_degree-1)

        self.W = MixedFunctionSpace((self.V[0], self.V[1]))
