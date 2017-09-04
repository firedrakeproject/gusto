from firedrake import Constant, FiniteElement, TensorProductElement, HDiv, \
    FunctionSpace, MixedFunctionSpace, interval, Function
from gusto.diagnostics import Diagnostics, Perturbation, \
    SteadyStateError
from gusto.output import Output

__all__ = ["State", "ShallowWaterState", "CompressibleEulerState", "IncompressibleEulerState", "AdvectionDiffusionState"]


class SpaceCreator(object):

    def __call__(self, name, mesh=None, family=None, degree=None):
        try:
            return getattr(self, name)
        except AttributeError:
            value = FunctionSpace(mesh, family, degree)
            setattr(self, name, value)
            return value


class FieldCreator(object):

    def __init__(self, fieldlist=None, xn=None, pickup=True):
        self.fields = []
        if fieldlist is not None:
            for name, func in zip(fieldlist, xn.split()):
                setattr(self, name, func)
                func.pickup = pickup
                func.rename(name)
                self.fields.append(func)

    def __call__(self, name, space=None, pickup=True):
        try:
            return getattr(self, name)
        except AttributeError:
            value = Function(space, name=name)
            setattr(self, name, value)
            value.pickup = pickup
            self.fields.append(value)
            return value

    def __iter__(self):
        return iter(self.fields)


class State(object):
    """
    Build a model state to keep the variables in, and specify parameters.

    :arg mesh: The :class:`Mesh` to use.
    :arg vertical_degree: integer, required for vertically extruded meshes.
    Specifies the degree for the pressure space in the vertical
    (the degrees for other spaces are inferred). Defaults to None.
    :arg horizontal_degree: integer, the degree for spaces in the horizontal
    (specifies the degree for the pressure space, other spaces are inferred)
    defaults to 1.
    :arg family: string, specifies the velocity space family to use.
    Options:
    "RT": The Raviart-Thomas family (default, recommended for quads)
    "BDM": The BDM family
    "BDFM": The BDFM family
    :arg output: class containing output parameters
    :arg diagnostics: class containing diagnostic methods
    :arg diagnostic_fields: list of diagnostic field classes
    """

    def __init__(self, mesh, fieldlist,
                 vertical_degree, horizontal_degree,
                 family,
                 output,
                 diagnostic_fields,
                 diagnostics):

        self.fieldlist = fieldlist
        self.diagnostics = diagnostics

        if output is None:
            raise RuntimeError("You must provide a directory name for dumping results")

        # The mesh
        self.mesh = mesh

        # Build the spaces
        self._build_spaces(fieldlist, mesh, vertical_degree, horizontal_degree, family)

        if fieldlist is not None:
            # Allocate state
            self._allocate_state()
            self.fields = FieldCreator(fieldlist, self.xn)
        else:
            self.fields = FieldCreator()

        #  Constant to hold current time
        self.t = Constant(0.0)

        if diagnostic_fields is not None:
            self.diagnostic_fields = diagnostic_fields
        else:
            self.diagnostic_fields = []

        if output.dumplist is None:
            dumplist = self.fieldlist + self.diagnostic_fields
        else:
            dumplist = output.dumplist + self.diagnostic_fields
        if output.dumplist_latlon is not None:
            dumplist_latlon = output.dumplist_latlon + self.diagnostic_fields
        self.output = Output(self, output, dumplist, dumplist_latlon)

    def setup_diagnostics(self, model):

        # add special case diagnostic fields
        for name in self.output.output_params.perturbation_fields:
            f = Perturbation(name)
            self.diagnostic_fields.append(f)

        for name in self.output.output_params.steady_state_error_fields:
            f = SteadyStateError(self, name)
            self.diagnostic_fields.append(f)

        for diagnostic in self.diagnostic_fields:
            print(diagnostic.name)
            diagnostic.setup(model)
            self.diagnostics.register(diagnostic.name)

    def initialise(self, initial_conditions):
        """
        Initialise state variables

        :arg initial_conditions: An iterable of pairs (field_name, pointwise_value)
        """
        for name, ic in initial_conditions:
            f_init = getattr(self.fields, name)
            f_init.assign(ic)
            f_init.rename(name)

    def set_reference_profiles(self, reference_profiles):
        """
        Initialise reference profiles

        :arg reference_profiles: An iterable of pairs (field_name, interpolatory_value)
        """
        for name, profile in reference_profiles:
            field = getattr(self.fields, name)
            ref = self.fields(name+'bar', field.function_space(), False)
            ref.interpolate(profile)

    def _build_spaces(self, fieldlist, mesh, vertical_degree, horizontal_degree, family):
        """
        Build:
        velocity space self.V2,
        pressure space self.V3,
        temperature space self.Vt,
        mixed function space self.W = (V2,V3,Vt)
        """

        self.spaces = SpaceCreator()
        if vertical_degree is not None:
            # horizontal base spaces
            cell = mesh._base_mesh.ufl_cell().cellname()
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

            V0 = self.spaces("HDiv", mesh, V2_elt)
            V1 = self.spaces("DG", mesh, V3_elt)
            V2 = self.spaces("HDiv_v", mesh, V2t_elt)

            self.Vv = self.spaces("Vv", mesh, V2v_elt)

            if fieldlist is not None:
                self.W = MixedFunctionSpace((V0, V1, V2))

        else:
            cell = mesh.ufl_cell().cellname()
            V1_elt = FiniteElement(family, cell, horizontal_degree+1)

            V0 = self.spaces("HDiv", mesh, V1_elt)
            V1 = self.spaces("DG", mesh, "DG", horizontal_degree)

            if fieldlist is not None:
                self.W = MixedFunctionSpace((V0, V1))

    def _allocate_state(self):
        """
        Construct Functions to store the state variables.
        """

        W = self.W
        self.xn = Function(W)
        self.xstar = Function(W)
        self.xp = Function(W)
        self.xnp1 = Function(W)
        self.xrhs = Function(W)
        self.xb = Function(W)  # store the old state for diagnostics
        self.dy = Function(W)


def ShallowWaterState(mesh,
                      horizontal_degree=1,
                      family="BDM",
                      output=None,
                      diagnostic_fields=None,
                      diagnostics=None):
    fieldlist = ['u', 'D']
    vertical_degree = None
    if diagnostics is None:
        diagnostics = Diagnostics(*fieldlist)

    return State(mesh, fieldlist,
                 vertical_degree, horizontal_degree, family,
                 output,
                 diagnostic_fields,
                 diagnostics)


def CompressibleEulerState(mesh, is_3d=False,
                           vertical_degree=1,
                           horizontal_degree=1,
                           family=None,
                           output=None,
                           diagnostic_fields=None,
                           diagnostics=None):
    fieldlist = ['u', 'rho', 'theta']
    if family is None:
        if is_3d:
            family = "RT"
        else:
            family = "CG"
    if diagnostics is None:
        diagnostics = Diagnostics(*fieldlist)

    return State(mesh, fieldlist,
                 vertical_degree, horizontal_degree, family,
                 output,
                 diagnostic_fields,
                 diagnostics)


def IncompressibleEulerState(mesh, is_3d=False,
                             vertical_degree=1,
                             horizontal_degree=1,
                             family=None,
                             output=None,
                             diagnostic_fields=None,
                             diagnostics=None):
    fieldlist = ['u', 'p', 'b']
    if family is None:
        if is_3d:
            family = "RT"
        else:
            family = "CG"
    if diagnostics is None:
        diagnostics = Diagnostics(*fieldlist)

    return State(mesh, fieldlist,
                 vertical_degree, horizontal_degree, family,
                 output,
                 diagnostic_fields,
                 diagnostics)


def AdvectionDiffusionState(mesh,
                            family,
                            horizontal_degree,
                            *,
                            vertical_degree=None,
                            output=None,
                            diagnostic_fields=None,
                            diagnostics=None):
    fieldlist = None
    return State(mesh, fieldlist,
                 vertical_degree, horizontal_degree, family,
                 output,
                 diagnostic_fields,
                 diagnostics)
