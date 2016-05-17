from __future__ import absolute_import
from os import path
import itertools
from collections import defaultdict
from functools import partial
import json
from dcore.diagnostics import Diagnostics
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
    :arg diagnostics: class containing diagnostic methods
    :arg fieldlist: list of prognostic field names

    """
    __metaclass__ = ABCMeta

    def __init__(self, mesh, vertical_degree=1, horizontal_degree=1,
                 family="RT", z=None, k=None, Omega=None,
                 timestepping=None,
                 output=None,
                 parameters=None,
                 diagnostics=None,
                 fieldlist=None,
                 diagnostic_fields=[]):

        self.z = z
        self.k = k
        self.Omega = Omega
        self.timestepping = timestepping
        self.output = output
        self.parameters = parameters
        if fieldlist is None:
            raise RuntimeError("You must provide a fieldlist containing the names of the prognostic fields")
        else:
            self.fieldlist = fieldlist
        if diagnostics is not None:
            self.diagnostics = diagnostics
        else:
            self.diagnostics = Diagnostics(*fieldlist)
        self.diagnostic_fields = diagnostic_fields

        # The mesh
        self.mesh = mesh

        # Build the spaces
        self._build_spaces(mesh, vertical_degree,
                           horizontal_degree, family)

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
        field_dict = {name: func for (name, func) in zip(self.fieldlist, funcs)}
        to_dump = []
        for name, f in field_dict.iteritems():
            if name in self.output.dumplist:
                to_dump.append(f)
            f.rename(name=name)
        for diagnostic in self.diagnostic_fields:
            to_dump.append(diagnostic(self))

        steady_state_dump_err = defaultdict(bool)
        steady_state_dump_err.update(self.output.steady_state_dump_err)
        for name, f, f_init in zip(self.fieldlist, funcs, self.x_init.split()):
            if steady_state_dump_err[name]:
                err = Function(f.function_space(), name=name+'err').assign(f-f_init)
                field_dict[name+"err"] = err
                self.diagnostics.register(name+"err")
                to_dump.append(err)

        meanfields = defaultdict(lambda: None)
        meanfields.update(self.output.meanfields)
        for name, meanfield in meanfields.iteritems():
            if meanfield is not None:
                field = field_dict[name]
                diff = Function(
                    field.function_space(),
                    name=field.name()+"_perturbation").assign(field - meanfield)
                self.diagnostics.register(name+"perturbation")
                field_dict[name+"perturbation"] = diff
                to_dump.append(diff)

        self.dumpdir = path.join("results", self.output.dirname)
        outfile = path.join(self.dumpdir, "field_output.pvd")
        if self.dumpfile is None:
            if path.exists(self.dumpdir):
                exit("results directory '%s' already exists" % self.dumpdir)
            self.dumpcount = itertools.count()
            self.dumpfile = File(outfile, project_output=self.output.project_fields)
            self.diagnostic_data = defaultdict(partial(defaultdict, list))

        if (next(self.dumpcount) % self.output.dumpfreq) == 0:

            self.dumpfile.write(*to_dump)

            for name in self.diagnostics.fields:
                data = self.diagnostics.l2(field_dict[name])
                self.diagnostic_data[name]["l2"].append(data)

    def diagnostic_dump(self):

        with open(path.join(self.dumpdir, "diagnostics.json"), "w") as f:
            f.write(json.dumps(self.diagnostic_data, indent=4))

    def initialise(self, initial_conditions):
        """
        Initialise state variables
        """

        for x, ic in zip(self.x_init.split(), initial_conditions):
            x.assign(ic)

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


class CompressibleState(State):

    def __init__(self, mesh, vertical_degree=1, horizontal_degree=1,
                 family="RT", z=None, k=None, Omega=None,
                 timestepping=None,
                 output=None,
                 parameters=None,
                 diagnostics=None,
                 fieldlist=None,
                 diagnostic_fields=[]):

        super(CompressibleState, self).__init__(mesh=mesh,
                                                vertical_degree=vertical_degree,
                                                horizontal_degree=horizontal_degree,
                                                family=family,
                                                z=z, k=k, Omega=Omega,
                                                timestepping=timestepping,
                                                output=output,
                                                parameters=parameters,
                                                diagnostics=diagnostics,
                                                fieldlist=fieldlist,
                                                diagnostic_fields=diagnostic_fields)

        # build the geopotential
        V = FunctionSpace(mesh, "CG", 1)
        self.Phi = Function(V).interpolate(Expression("pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)"))
        self.Phi *= parameters.g

        if self.k is None:
            # build the vertical normal
            w = TestFunction(self.Vv)
            u = TrialFunction(self.Vv)
            self.k = Function(self.Vv)
            n = FacetNormal(self.mesh)
            krhs = -div(w)*self.z*dx + inner(w,n)*self.z*ds_tb
            klhs = inner(w,u)*dx
            solve(klhs == krhs, self.k)

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

        V1_elt = FiniteElement(family, cell, horizontal_degree+1)

        self.V = [0,0]
        self.V[0] = FunctionSpace(mesh,V1_elt)
        self.V[1] = FunctionSpace(mesh,"DG",horizontal_degree)

        self.W = MixedFunctionSpace((self.V[0], self.V[1]))
