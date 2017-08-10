from abc import ABCMeta, abstractproperty
from collections import defaultdict
from firedrake import SpatialCoordinate, sqrt, inner, interpolate, CellNormal, cross, Constant, as_vector
from gusto.advection import SSPRK3, ThetaMethod
from gusto.configuration import ShallowWaterParameters, CompressibleParameters
from gusto.forcing import ShallowWaterForcing, CompressibleForcing
from gusto.linear_solvers import ShallowWaterSolver, CompressibleSolver
from gusto.state import State
from gusto.transport_equation import VectorInvariant, AdvectionEquation, SUPGAdvection


class Model(object, metaclass=ABCMeta):

    eqn_opts = defaultdict(dict)

    def __init__(self,
                 mesh,
                 timestepping=None,
                 output=None,
                 parameters=None,
                 diagnostics=None,
                 diagnostic_fields=None):

        # store timestepping class
        self.timestepping = timestepping

        # update physical parameters
        if parameters is not None:
            for k, v in parameters.items():
                self.parameters.__setattr__(k, v)

        # create state
        self.state = State(mesh=mesh,
                           vertical_degree=self.vertical_degree,
                           horizontal_degree=self.horizontal_degree,
                           family=self.family,
                           fieldlist=self.fieldlist,
                           output=output,
                           diagnostics=diagnostics,
                           diagnostic_fields=diagnostic_fields)

        # figure out if we're on a sphere
        try:
            self.on_sphere = (mesh._base_mesh.geometric_dimension() == 3 and mesh._base_mesh.topological_dimension() == 2)
        except AttributeError:
            self.on_sphere = (mesh.geometric_dimension() == 3 and mesh.topological_dimension() == 2)

        #  build the vertical normal and define perp for 2d geometries
        dim = mesh.topological_dimension()
        if self.on_sphere:
            x = SpatialCoordinate(mesh)
            R = sqrt(inner(x, x))
            self.state.k = interpolate(x/R, mesh.coordinates.function_space())
            if dim == 2:
                outward_normals = CellNormal(mesh)
                self.state.perp = lambda u: cross(outward_normals, u)
        else:
            kvec = [0.0]*dim
            kvec[dim-1] = 1.0
            self.state.k = Constant(kvec)
            if dim == 2:
                self.state.perp = lambda u: as_vector([-u[1], u[0]])

    @abstractproperty
    def family(self):
        pass

    @abstractproperty
    def vertical_degree(self):
        pass

    @abstractproperty
    def horizontal_degree(self):
        pass

    @abstractproperty
    def fieldlist(self):
        pass

    @abstractproperty
    def parameters(self):
        pass

    @abstractproperty
    def linear_solver(self):
        pass

    @abstractproperty
    def forcing(self):
        pass

    @abstractproperty
    def default_field_equations(self):
        pass

    @abstractproperty
    def default_advection_schemes(self):
        pass

    def setup(self, linear_solver=None, forcing=None, field_equations=None,
              advection_schemes=None):

        dt = self.timestepping.dt
        alpha = self.timestepping.alpha
        # if linear solver class is passed in then add it to model,
        # else instantiate the default linear solver class
        if linear_solver is not None:
            self.linear_solver = linear_solver
        else:
            self.linear_solver = self.linear_solver(self.state, self.parameters, beta=dt*alpha)

        state = self.state

        # If no field_equations dict has been passed in then create an
        # empty dict to iterate over. If field_equations have been
        # passed in then update the defaults
        if field_equations is None:
            instantiated = []
            self.field_equations = self.default_field_equations.copy()
        else:
            # list of field equations that have already been instantiated
            instantiated = [fname for fname in field_equations.keys()]
            f = self.default_field_equations.copy()
            f.update(field_equations)
            self.field_equations = f
        # instantiate the default field equation classes if user has
        # not passed in another option
        for fname, eqn in self.field_equations.items():
            if fname not in instantiated:
                field = state.fields(fname)
                space = field.function_space()
                self.field_equations[fname] = eqn(state, space,
                                                  **self.eqn_opts[fname])

        # if forcing class is passed in then add it to model, else
        # instantiate the default forcing class
        if forcing is not None:
            self.forcing = forcing
        else:
            self.forcing = self.forcing(state, self.parameters)

        # if no advection_schemes dict has been passed in then create an
        # empty dict to iterate over
        if advection_schemes is None:
            instantiated = []
            self.advection_schemes = self.default_advection_schemes.copy()
        else:
            # list of advection schemes that have already been instantiated
            instantiated = [fname for fname in advection_schemes.keys()]
            a = self.default_advection_schemes.copy()
            a.update(advection_schemes)
            self.advection_schemes = a
        # instantiate the default advection scheme classes if user has
        # not passed in another option
        for fname, scheme in self.advection_schemes.items():
            if fname not in instantiated:
                self.advection_schemes[fname] = scheme(
                    state.fields(fname), dt, self.field_equations[fname])

        # set up advected fields list of tuples (field_name, advection scheme)
        self.advected_fields = [
            (fname, scheme) for fname, scheme in self.advection_schemes.items()
        ]

        self.diffused_fields = None
        self.physics_list = None
        self.mu = None


class ShallowWaterModel(Model):

    family = "BDM"
    vertical_degree = None
    horizontal_degree = 1
    fieldlist = ["u", "D"]
    parameters = ShallowWaterParameters()
    linear_solver = ShallowWaterSolver
    forcing = ShallowWaterForcing
    default_advection_schemes = {"u": ThetaMethod, "D": SSPRK3}
    default_field_equations = {"u": VectorInvariant, "D": AdvectionEquation}

    def __init__(self,
                 mesh,
                 timestepping=None,
                 output=None,
                 parameters=None,
                 diagnostics=None,
                 diagnostic_fields=None):

        super().__init__(mesh,
                         timestepping,
                         output,
                         parameters,
                         diagnostics,
                         diagnostic_fields)
        self.eqn_opts["D"] = {"ibp": "once", "equation_form": "continuity"}


class CompressibleEulerModel(Model):

    family = "CG"
    vertical_degree = 1
    horizontal_degree = 1
    fieldlist = ["u", "rho", "theta"]
    parameters = CompressibleParameters()
    linear_solver = CompressibleSolver
    forcing = CompressibleForcing
    default_advection_schemes = {"u": ThetaMethod, "rho": SSPRK3, "theta": SSPRK3}
    default_field_equations = {"u": VectorInvariant, "rho": AdvectionEquation,
                               "theta": SUPGAdvection}

    def __init__(self,
                 mesh,
                 timestepping=None,
                 output=None,
                 parameters=None,
                 diagnostics=None,
                 diagnostic_fields=None):

        super().__init__(mesh,
                         timestepping,
                         output,
                         parameters,
                         diagnostics,
                         diagnostic_fields)
        self.eqn_opts["rho"] = {"ibp": "once", "equation_form": "continuity"}
        self.eqn_opts["theta"] = {"dt": timestepping.dt, 
                                  "supg_params": {"dg_direction": "horizontal"}}
