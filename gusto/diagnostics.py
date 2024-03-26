"""Common diagnostic fields."""

from firedrake import assemble, dot, dx, Function, sqrt, \
    TestFunction, TrialFunction, Constant, grad, inner, curl, \
    LinearVariationalProblem, LinearVariationalSolver, FacetNormal, \
    ds_b, ds_v, ds_t, dS_h, dS_v, ds, dS, div, avg, jump, pi, \
    TensorFunctionSpace, SpatialCoordinate, as_vector, \
    Projector, Interpolator, FunctionSpace, FiniteElement, \
    TensorProductElement
from firedrake.assign import Assigner

from abc import ABCMeta, abstractmethod, abstractproperty
import gusto.thermodynamics as tde
from gusto.coord_transforms import rotated_lonlatr_vectors
from gusto.recovery import Recoverer, BoundaryMethod
from gusto.equations import CompressibleEulerEquations
from gusto.active_tracers import TracerVariableType, Phases
from gusto.logging import logger
from gusto.kernels import MinKernel, MaxKernel
import numpy as np

__all__ = ["Diagnostics", "CourantNumber", "Gradient", "XComponent", "YComponent",
           "ZComponent", "MeridionalComponent", "ZonalComponent", "RadialComponent",
           "RichardsonNumber", "Energy", "KineticEnergy", "ShallowWaterKineticEnergy",
           "ShallowWaterPotentialEnergy", "ShallowWaterPotentialEnstrophy",
           "CompressibleKineticEnergy", "Exner", "Sum", "Difference", "SteadyStateError",
           "Perturbation", "Theta_e", "InternalEnergy", "PotentialEnergy",
           "ThermodynamicKineticEnergy", "Dewpoint", "Temperature", "Theta_d",
           "RelativeHumidity", "Pressure", "Exner_Vt", "HydrostaticImbalance", "Precipitation",
           "PotentialVorticity", "RelativeVorticity", "AbsoluteVorticity", "Divergence",
           "BruntVaisalaFrequencySquared", "TracerDensity"]


class Diagnostics(object):
    """
    Stores all diagnostic fields, and controls global diagnostics computation.

    This object stores the diagnostic fields to be output, and the computation
    of global values from them (such as global maxima or norms).
    """

    available_diagnostics = ["min", "max", "rms", "l2", "total"]

    def __init__(self, *fields):
        """
        Args:
            *fields: list of :class:`Function` objects of fields to be output.
        """

        self.fields = list(fields)

    def register(self, *fields):
        """
        Registers diagnostic fields for outputting.

        Args:
            *fields: list of :class:`Function` objects of fields to be output.
        """

        fset = set(self.fields)
        for f in fields:
            if f not in fset:
                self.fields.append(f)

    @staticmethod
    def min(f):
        """
        Finds the global minimum DoF value of a field.

        Args:
            f (:class:`Function`): field to compute diagnostic for.
        """
        min_kernel = MinKernel()
        return min_kernel.apply(f)

    @staticmethod
    def max(f):
        """
        Finds the global maximum DoF value of a field.

        Args:
            f (:class:`Function`): field to compute diagnostic for.
        """
        max_kernel = MaxKernel()
        return max_kernel.apply(f)

    @staticmethod
    def rms(f):
        """
        Calculates the root-mean-square of a field.

        Args:
            f (:class:`Function`): field to compute diagnostic for.
        """

        area = assemble(1*dx(domain=f.ufl_domain()))
        return sqrt(assemble(inner(f, f)*dx)/area)

    @staticmethod
    def l2(f):
        """
        Calculates the L2 norm of a field.

        Args:
            f (:class:`Function`): field to compute diagnostic for.
        """

        return sqrt(assemble(inner(f, f)*dx))

    @staticmethod
    def total(f):
        """
        Calculates the total of a field. Only applicable for fields with
        scalar-values.

        Args:
            f (:class:`Function`): field to compute diagnostic for.
        """

        if len(f.ufl_shape) == 0:
            return assemble(f * dx)
        else:
            pass


class DiagnosticField(object, metaclass=ABCMeta):
    """Base object to represent diagnostic fields for outputting."""
    def __init__(self, space=None, method='interpolate', required_fields=()):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
            required_fields (tuple, optional): tuple of names of the fields that
                are required for the computation of this diagnostic field.
                Defaults to ().
        """

        assert method in ['interpolate', 'project', 'solve', 'assign'], \
            f'Invalid evaluation method {self.method} for diagnostic {self.name}'

        self._initialised = False
        self.required_fields = required_fields
        self.space = space
        self.method = method
        self.expr = None
        self.to_dump = True

        # Property to allow graceful failures if solve method not valid
        if not hasattr(self, "solve_implemented"):
            self.solve_implemented = False

        if method == 'solve' and not self.solve_implemented:
            raise NotImplementedError(f'Solve method has not been implemented for diagnostic {self.name}')

    @abstractproperty
    def name(self):
        """The name of this diagnostic field"""
        pass

    @abstractmethod
    def setup(self, domain, state_fields, space=None):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
            space (:class:`FunctionSpace`, optional): the function space for the
                diagnostic field to be computed in. Defaults to None, in which
                case the space will be DG0.
        """

        if not self._initialised:
            if self.space is None:
                if space is None:
                    if not hasattr(domain.spaces, "DG0"):
                        space = domain.spaces.create_space("DG0", "DG", 0)
                    else:
                        space = domain.spaces("DG0")
                self.space = space
            else:
                space = self.space

            # Add space to domain
            assert space.name is not None, \
                f'Diagnostics {self.name} is using a function space which does not have a name'
            if not hasattr(domain.spaces, space.name):
                domain.spaces.add_space(space.name, space)

            self.field = state_fields(self.name, space=space, dump=self.to_dump, pick_up=False)

            if self.method != 'solve':
                assert self.expr is not None, \
                    f"The expression for diagnostic {self.name} has not been specified"

            # Solve method must be declared in diagnostic's own setup routine
            if self.method == 'interpolate':
                self.evaluator = Interpolator(self.expr, self.field)
            elif self.method == 'project':
                self.evaluator = Projector(self.expr, self.field)
            elif self.method == 'assign':
                self.evaluator = Assigner(self.field, self.expr)

            self._initialised = True

    def compute(self):
        """Compute the diagnostic field from the current state."""

        logger.debug(f'Computing diagnostic {self.name} with {self.method} method')

        if self.method == 'interpolate':
            self.evaluator.interpolate()
        elif self.method == 'assign':
            self.evaluator.assign()
        elif self.method == 'project':
            self.evaluator.project()
        elif self.method == 'solve':
            self.evaluator.solve()

    def __call__(self):
        """Return the diagnostic field computed from the current state."""
        self.compute()
        return self.field


class CourantNumber(DiagnosticField):
    """Dimensionless Courant number diagnostic field."""
    name = "CourantNumber"

    def __init__(self, velocity='u', component='whole', name=None, to_dump=True,
                 space=None, method='interpolate', required_fields=()):
        """
        Args:
            velocity (str or :class:`ufl.Expr`, optional): the velocity field to
                take the Courant number of. Can be a string referring to an
                existing field, or an expression. If it is an expression, the
                name argument is required. Defaults to 'u'.
            component (str, optional): the component of the velocity to use for
                calculating the Courant number. Valid values are "whole",
                "horizontal" or "vertical". Defaults to "whole".
            name (str, optional): the name to append to "CourantNumber" to form
                the name of this diagnostic. This argument must be provided if
                the velocity is an expression (rather than a string). Defaults
                to None.
            to_dump (bool, optional): whether this diagnostic should be dumped.
                Defaults to True.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
            required_fields (tuple, optional): tuple of names of the fields that
                are required for the computation of this diagnostic field.
                Defaults to ().
        """
        if component not in ["whole", "horizontal", "vertical"]:
            raise ValueError(f'component arg {component} not valid. Allowed '
                             + 'values are "whole", "horizontal" and "vertical"')
        self.component = component

        # Work out whether to take Courant number from field or expression
        if type(velocity) is str:
            # Default name should just be CourantNumber
            if velocity == 'u':
                self.name = 'CourantNumber'
            elif name is None:
                self.name = 'CourantNumber_'+velocity
            else:
                self.name = 'CourantNumber_'+name
            if component != 'whole':
                self.name += '_'+component
        else:
            if name is None:
                raise ValueError('CourantNumber diagnostic: if provided '
                                 + 'velocity is an expression then the name '
                                 + 'argument must be provided')
            self.name = 'CourantNumber_'+name

        self.velocity = velocity
        super().__init__(space=space, method=method, required_fields=required_fields)

        # Done after super init to ensure that it is not always set to True
        self.to_dump = to_dump

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        V = FunctionSpace(domain.mesh, "DG", 0)
        test = TestFunction(V)
        cell_volume = Function(V)
        self.cell_flux = Function(V)

        # Calculate cell volumes
        One = Function(V).assign(1)
        assemble(One*test*dx, tensor=cell_volume)

        # Get the velocity that is being used
        if type(self.velocity) is str:
            u = state_fields(self.velocity)
        else:
            u = self.velocity

        # Determine the component of the velocity
        if self.component == "whole":
            u_expr = u
        elif self.component == "vertical":
            u_expr = dot(u, domain.k)*domain.k
        elif self.component == "horizontal":
            u_expr = u - dot(u, domain.k)*domain.k

        # Work out which facet integrals to use
        if domain.mesh.extruded:
            dS_calc = dS_v + dS_h
            ds_calc = ds_v + ds_t + ds_b
        else:
            dS_calc = dS
            ds_calc = ds

        # Set up form for DG flux
        n = FacetNormal(domain.mesh)
        un = 0.5*(inner(-u_expr, n) + abs(inner(-u_expr, n)))
        self.cell_flux_form = 2*avg(un*test)*dS_calc + un*test*ds_calc

        # Final Courant number expression
        self.expr = self.cell_flux * domain.dt / cell_volume

        super().setup(domain, state_fields)

    def compute(self):
        """Compute the diagnostic field from the current state."""

        assemble(self.cell_flux_form, tensor=self.cell_flux)
        super().compute()


class Gradient(DiagnosticField):
    """Diagnostic for computing the gradient of fields."""
    def __init__(self, name, space=None, method='solve'):
        """
        Args:
            name (str): name of the field to compute the gradient of.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'solve'.
        """
        self.fname = name
        self.solve_implemented = True
        super().__init__(space=space, method=method, required_fields=(name,))

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_gradient"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        f = state_fields(self.fname)

        mesh_dim = domain.mesh.geometric_dimension()
        try:
            field_dim = state_fields(self.fname).ufl_shape[0]
        except IndexError:
            field_dim = 1
        shape = (mesh_dim, ) * field_dim
        space = TensorFunctionSpace(domain.mesh, "CG", 1, shape=shape, name=f'Tensor{field_dim}_CG1')

        if self.method != 'solve':
            self.expr = grad(f)

        super().setup(domain, state_fields, space=space)

        # Set up problem now that self.field has been set up
        if self.method == 'solve':
            test = TestFunction(space)
            trial = TrialFunction(space)
            n = FacetNormal(domain.mesh)
            a = inner(test, trial)*dx
            L = -inner(div(test), f)*dx
            if space.extruded:
                L += dot(dot(test, n), f)*(ds_t + ds_b)
            prob = LinearVariationalProblem(a, L, self.field)
            self.evaluator = LinearVariationalSolver(prob)


class Divergence(DiagnosticField):
    """Diagnostic for computing the divergence of vector-valued fields."""
    def __init__(self, name='u', space=None, method='interpolate'):
        """
        Args:
            name (str, optional): name of the field to compute the gradient of.
                Defaults to 'u', in which case this takes the divergence of the
                wind field.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case the default space is the domain's DG space.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.fname = name
        super().__init__(space=space, method=method, required_fields=(self.fname,))

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_divergence"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        f = state_fields(self.fname)
        self.expr = div(f)
        space = domain.spaces("DG")
        super().setup(domain, state_fields, space=space)


class VectorComponent(DiagnosticField):
    """Base diagnostic for orthogonal components of vector-valued fields."""
    def __init__(self, name, space=None, method='interpolate'):
        """
        Args:
            name (str): name of the field to compute the component of.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case the default space is the domain's DG space.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.fname = name
        super().__init__(space=space, method=method, required_fields=(name,))

    def setup(self, domain, state_fields, unit_vector):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
            unit_vector (:class:`ufl.Expr`): the unit vector to extract the
                component for. This assumes an orthogonal coordinate system.
        """
        f = state_fields(self.fname)
        self.expr = dot(f, unit_vector)
        super().setup(domain, state_fields)


class XComponent(VectorComponent):
    """The geocentric Cartesian x-component of a vector-valued field."""
    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_x"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        dim = domain.mesh.topological_dimension()
        e_x = as_vector([Constant(1.0)]+[Constant(0.0)]*(dim-1))
        super().setup(domain, state_fields, e_x)


class YComponent(VectorComponent):
    """The geocentric Cartesian y-component of a vector-valued field."""
    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_y"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        assert domain.metadata['domain_type'] not in ['interval', 'vertical_slice'], \
            f'Y-component diagnostic cannot be used with domain {domain.metadata["domain_type"]}'
        dim = domain.mesh.topological_dimension()
        e_y = as_vector([Constant(0.0), Constant(1.0)]+[Constant(0.0)]*(dim-2))
        super().setup(domain, state_fields, e_y)


class ZComponent(VectorComponent):
    """The geocentric Cartesian z-component of a vector-valued field."""
    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_z"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        assert domain.metadata['domain_type'] not in ['interval', 'plane'], \
            f'Z-component diagnostic cannot be used with domain {domain.metadata["domain_type"]}'
        dim = domain.mesh.topological_dimension()
        e_x = as_vector([Constant(0.0)]*(dim-1)+[Constant(1.0)])
        super().setup(domain, state_fields, e_x)


class SphericalComponent(VectorComponent):
    """Base diagnostic for computing spherical-polar components of fields."""
    def __init__(self, name, rotated_pole=None, space=None, method='interpolate'):
        """
        Args:
            name (str): name of the field to compute the component of.
            rotated_pole (tuple of floats, optional): a tuple of floats
                (lon, lat) of the new pole, in the original coordinate system.
                The longitude and latitude must be expressed in radians.
                Defaults to None, corresponding to a pole of (0, pi/2).
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case the default space is the domain's DG space.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.rotated_pole = (0.0, pi/2) if rotated_pole is None else rotated_pole
        super().__init__(name=name, space=space, method=method)

    def _check_args(self, domain, field):
        """
        Checks the validity of the domain and field for taking the spherical
        component diagnostic.

        Args:
            domain (:class:`Domain`): the model's domain object.
            field (:class:`Function`): the field to take the component of.
        """

        # check geometric dimension is 3D
        if domain.mesh.geometric_dimension() != 3:
            raise ValueError('Spherical components only work when the geometric dimension is 3!')

        if np.prod(field.ufl_shape) != 3:
            raise ValueError('Components can only be found of a vector function space in 3D.')


class MeridionalComponent(SphericalComponent):
    """The meridional component of a vector-valued field."""
    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_meridional"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        f = state_fields(self.fname)
        self._check_args(domain, f)
        xyz = SpatialCoordinate(domain.mesh)
        _, e_lat, _ = rotated_lonlatr_vectors(xyz, self.rotated_pole)
        super().setup(domain, state_fields, e_lat)


class ZonalComponent(SphericalComponent):
    """The zonal component of a vector-valued field."""
    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_zonal"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        f = state_fields(self.fname)
        self._check_args(domain, f)
        xyz = SpatialCoordinate(domain.mesh)
        e_lon, _, _ = rotated_lonlatr_vectors(xyz, self.rotated_pole)
        super().setup(domain, state_fields, e_lon)


class RadialComponent(SphericalComponent):
    """The radial component of a vector-valued field."""
    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_radial"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        f = state_fields(self.fname)
        self._check_args(domain, f)
        xyz = SpatialCoordinate(domain.mesh)
        _, _, e_r = rotated_lonlatr_vectors(xyz, self.rotated_pole)
        super().setup(domain, state_fields, e_r)


class RichardsonNumber(DiagnosticField):
    """Dimensionless Richardson number diagnostic field."""
    name = "RichardsonNumber"

    def __init__(self, density_field, factor=1., space=None, method='interpolate'):
        u"""
        Args:
            density_field (str): the name of the density field.
            factor (float, optional): a factor to multiply the Brunt-Väisälä
                frequency by. Defaults to 1.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        super().__init__(space=space, method=method, required_fields=(density_field, "u_gradient"))
        self.density_field = density_field
        self.factor = Constant(factor)

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        rho_grad = self.density_field+"_gradient"
        grad_density = state_fields(rho_grad)
        gradu = state_fields("u_gradient")

        denom = 0.
        z_dim = domain.mesh.geometric_dimension() - 1
        u_dim = state_fields("u").ufl_shape[0]
        for i in range(u_dim-1):
            denom += gradu[i, z_dim]**2
        Nsq = self.factor*grad_density[z_dim]
        self.expr = Nsq/denom
        super().setup(domain, state_fields)


# TODO: unify all energy diagnostics -- should be based on equation
class Energy(DiagnosticField):
    """Base diagnostic field for computing energy density fields."""
    def kinetic(self, u, factor=None):
        """
        Computes a kinetic energy term.

        Args:
            u (:class:`ufl.Expr`): the velocity variable.
            factor (:class:`ufl.Expr`, optional): factor to multiply the term by
                (e.g. a density variable). Defaults to None.

        Returns:
            :class:`ufl.Expr`: the kinetic energy term
        """
        if factor is not None:
            energy = 0.5*factor*dot(u, u)
        else:
            energy = 0.5*dot(u, u)
        return energy


class KineticEnergy(Energy):
    """Diagnostic kinetic energy density."""
    name = "KineticEnergy"

    def __init__(self, space=None, method='interpolate'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        super().__init__(space=space, method=method, required_fields=("u"))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields("u")
        self.expr = self.kinetic(u)
        super().setup(domain, state_fields)


class ShallowWaterKineticEnergy(Energy):
    """Diagnostic shallow-water kinetic energy density."""
    name = "ShallowWaterKineticEnergy"

    def __init__(self, space=None, method='interpolate'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        super().__init__(space=space, method=method, required_fields=("D", "u"))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields("u")
        D = state_fields("D")
        self.expr = self.kinetic(u, D)
        super().setup(domain, state_fields)


class ShallowWaterPotentialEnergy(Energy):
    """Diagnostic shallow-water potential energy density."""
    name = "ShallowWaterPotentialEnergy"

    def __init__(self, parameters, space=None, method='interpolate'):
        """
        Args:
            parameters (:class:`ShallowWaterParameters`): the configuration
                object containing the physical parameters for this equation.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.parameters = parameters
        super().__init__(space=space, method=method, required_fields=("D"))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        g = self.parameters.g
        D = state_fields("D")
        self.expr = 0.5*g*D**2
        super().setup(domain, state_fields)


class ShallowWaterPotentialEnstrophy(DiagnosticField):
    """Diagnostic (dry) compressible kinetic energy density."""
    def __init__(self, base_field_name="PotentialVorticity", space=None,
                 method='interpolate'):
        """
        Args:
            base_field_name (str, optional): the base potential vorticity field
                to compute the enstrophy from. Defaults to "PotentialVorticity".
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        base_enstrophy_names = ["PotentialVorticity", "RelativeVorticity", "AbsoluteVorticity"]
        if base_field_name not in base_enstrophy_names:
            raise ValueError(
                f"Don't know how to compute enstrophy with base_field_name={base_field_name};"
                + f"base_field_name should be one of {base_enstrophy_names}")
        # Work out required fields
        if base_field_name in ["PotentialVorticity", "AbsoluteVorticity"]:
            required_fields = (base_field_name, "D")
        elif base_field_name == "RelativeVorticity":
            required_fields = (base_field_name, "D", "coriolis")
        else:
            raise NotImplementedError(f'Enstrophy with vorticity {base_field_name} not implemented')

        super().__init__(space=space, method=method, required_fields=required_fields)
        self.base_field_name = base_field_name

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        base_name = "SWPotentialEnstrophy"
        return "_from_".join((base_name, self.base_field_name))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        if self.base_field_name == "PotentialVorticity":
            pv = state_fields("PotentialVorticity")
            D = state_fields("D")
            self.expr = 0.5*pv**2*D
        elif self.base_field_name == "RelativeVorticity":
            zeta = state_fields("RelativeVorticity")
            D = state_fields("D")
            f = state_fields("coriolis")
            self.expr = 0.5*(zeta + f)**2/D
        elif self.base_field_name == "AbsoluteVorticity":
            zeta_abs = state_fields("AbsoluteVorticity")
            D = state_fields("D")
            self.expr = 0.5*(zeta_abs)**2/D
        else:
            raise NotImplementedError(f'Enstrophy with {self.base_field_name} not implemented')
        super().setup(domain, state_fields)


class CompressibleKineticEnergy(Energy):
    """Diagnostic (dry) compressible kinetic energy density."""
    name = "CompressibleKineticEnergy"

    def __init__(self, space=None, method='interpolate'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        super().__init__(space=space, method=method, required_fields=("rho", "u"))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field
        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields("u")
        rho = state_fields("rho")
        self.expr = self.kinetic(u, rho)
        super().setup(domain, state_fields)


class Exner(DiagnosticField):
    """The diagnostic Exner pressure field."""
    def __init__(self, parameters, reference=False, space=None, method='interpolate'):
        """
        Args:
            parameters (:class:`CompressibleParameters`): the configuration
                object containing the physical parameters for this equation.
            reference (bool, optional): whether to compute the reference Exner
                pressure field or not. Defaults to False.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.parameters = parameters
        self.reference = reference
        if reference:
            self.rho_name = "rho_bar"
            self.theta_name = "theta_bar"
        else:
            self.rho_name = "rho"
            self.theta_name = "theta"
        super().__init__(space=space, method=method, required_fields=(self.rho_name, self.theta_name))

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        if self.reference:
            return "Exner_bar"
        else:
            return "Exner"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        rho = state_fields(self.rho_name)
        theta = state_fields(self.theta_name)
        self.expr = tde.exner_pressure(self.parameters, rho, theta)
        super().setup(domain, state_fields)


class BruntVaisalaFrequencySquared(DiagnosticField):
    """The diagnostic for the Brunt-Väisälä frequency."""
    name = "Brunt-Vaisala_squared"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.parameters = equations.parameters
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['theta']
            if equations.active_tracers is not None and len(equations.active_tracers) > 1:
                # TODO: I think theta here should be theta_e, which would be
                # easiest if this is a ThermodynamicDiagnostic. But in the dry
                # case, our numerical theta_e does not reduce to the numerical
                # dry theta
                raise NotImplementedError(
                    'Brunt-Vaisala diagnostic not implemented for moist equations')
        else:
            raise NotImplementedError(
                f'Brunt-Vaisala diagnostic not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=tuple(required_fields))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        theta = state_fields('theta')
        self.expr = self.parameters.g/theta * dot(domain.k, grad(theta))
        super().setup(domain, state_fields)


class Sum(DiagnosticField):
    """Base diagnostic for computing the sum of two fields."""
    def __init__(self, field_name1, field_name2):
        """
        Args:
            field_name1 (str): the name of one field to be added.
            field_name2 (str): the name of the other field to be added.
        """
        super().__init__(method='assign', required_fields=(field_name1, field_name2))
        self.field_name1 = field_name1
        self.field_name2 = field_name2

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.field_name1+"_plus_"+self.field_name2

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        field1 = state_fields(self.field_name1)
        field2 = state_fields(self.field_name2)
        space = field1.function_space()
        self.expr = field1 + field2
        super().setup(domain, state_fields, space=space)


class Difference(DiagnosticField):
    """Base diagnostic for calculating the difference between two fields."""
    def __init__(self, field_name1, field_name2):
        """
        Args:
            field_name1 (str): the name of the field to be subtracted from.
            field_name2 (str): the name of the field to be subtracted.
        """
        super().__init__(method='assign', required_fields=(field_name1, field_name2))
        self.field_name1 = field_name1
        self.field_name2 = field_name2

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.field_name1+"_minus_"+self.field_name2

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        field1 = state_fields(self.field_name1)
        field2 = state_fields(self.field_name2)
        self.expr = field1 - field2
        space = field1.function_space()
        super().setup(domain, state_fields, space=space)


class SteadyStateError(Difference):
    """Base diagnostic for computing the steady-state error in a field."""
    def __init__(self, name):
        """
        Args:
            name (str): name of the field to take the steady-state error of.
        """
        self.field_name1 = name
        self.field_name2 = name+'_init'
        DiagnosticField.__init__(self, method='assign', required_fields=(name, self.field_name2))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        # Check if initial field already exists -- otherwise needs creating
        if not hasattr(state_fields, self.field_name2):
            field1 = state_fields(self.field_name1)
            field2 = state_fields(self.field_name2, space=field1.function_space(),
                                  pick_up=True, dump=False)
            # Attach state fields to self so that we can pick it up in compute
            self.state_fields = state_fields
            # The initial value for fields may not have already been set yet so we
            # postpone setting it until the compute method is called
            self.init_field_set = False
        else:
            field1 = state_fields(self.field_name1)
            field2 = state_fields(self.field_name2, space=field1.function_space(),
                                  pick_up=True, dump=False)
            # By default set this new field to the current value
            # This may be overwritten if picking up from a checkpoint
            field2.assign(field1)
            self.state_fields = state_fields
            self.init_field_set = True

        super().setup(domain, state_fields)

    def compute(self):
        # The first time the compute method is called we set the initial field.
        # We do not want to do this if picking up from a checkpoint
        if not self.init_field_set:
            # Set initial field
            full_field = self.state_fields(self.field_name1)
            init_field = self.state_fields(self.field_name2)
            init_field.assign(full_field)

            self.init_field_set = True

        super().compute()

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.field_name1+"_error"


class Perturbation(Difference):
    """Base diagnostic for computing perturbations from a reference profile."""
    def __init__(self, name):
        """
        Args:
            name (str): name of the field to take the perturbation of.
        """
        self.field_name1 = name
        self.field_name2 = name+'_bar'
        DiagnosticField.__init__(self, method='assign', required_fields=(name, self.field_name2))

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.field_name1+"_perturbation"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        # Check if initial field already exists -- otherwise needs creating
        if not hasattr(state_fields, self.field_name2):
            field1 = state_fields(self.field_name1)
            _ = state_fields(self.field_name2, space=field1.function_space(),
                             pick_up=True, dump=False)

        super().setup(domain, state_fields)


# TODO: unify thermodynamic diagnostics
class ThermodynamicDiagnostic(DiagnosticField):
    """Base thermodynamic diagnostic field, computing many common fields."""

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.equations = equations
        self.parameters = equations.parameters
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['rho', 'theta']
            if equations.active_tracers is not None:
                for active_tracer in equations.active_tracers:
                    if active_tracer.chemical == 'H2O':
                        required_fields.append(active_tracer.name)
        else:
            raise NotImplementedError(f'Thermodynamic diagnostics not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=tuple(required_fields))

    def _setup_thermodynamics(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        self.Vtheta = domain.spaces('theta')
        h_deg = self.Vtheta.ufl_element().degree()[0]
        v_deg = self.Vtheta.ufl_element().degree()[1]-1
        boundary_method = BoundaryMethod.extruded if (v_deg == 0 and h_deg == 0) else None

        # Extract all fields
        self.rho = state_fields("rho")
        self.theta = state_fields("theta")
        # Rho must be averaged to Vtheta
        self.rho_averaged = Function(self.Vtheta)
        self.recoverer = Recoverer(self.rho, self.rho_averaged, boundary_method=boundary_method)

        zero_expr = Constant(0.0)*self.theta
        self.r_v = zero_expr  # Water vapour
        self.r_l = zero_expr  # Liquid water
        self.r_t = zero_expr  # All water mixing ratios
        for active_tracer in self.equations.active_tracers:
            if active_tracer.chemical == "H2O":
                if active_tracer.variable_type != TracerVariableType.mixing_ratio:
                    raise NotImplementedError('Only mixing ratio tracers are implemented')
                if active_tracer.phase == Phases.gas:
                    self.r_v += state_fields(active_tracer.name)
                elif active_tracer.phase == Phases.liquid:
                    self.r_l += state_fields(active_tracer.name)
                self.r_t += state_fields(active_tracer.name)

        # Store the most common expressions
        self.exner = tde.exner_pressure(self.parameters, self.rho_averaged, self.theta)
        self.T = tde.T(self.parameters, self.theta, self.exner, r_v=self.r_v)
        self.p = tde.p(self.parameters, self.exner)

    def compute(self):
        """Compute the thermodynamic diagnostic."""
        self.recoverer.project()
        super().compute()


class Theta_e(ThermodynamicDiagnostic):
    """The moist equivalent potential temperature diagnostic field."""
    name = "Theta_e"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = tde.theta_e(self.parameters, self.T, self.p, self.r_v, self.r_t)
        super().setup(domain, state_fields, space=self.Vtheta)


class InternalEnergy(ThermodynamicDiagnostic):
    """The moist compressible internal energy density."""
    name = "InternalEnergy"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = tde.internal_energy(self.parameters, self.rho_averaged, self.T, r_v=self.r_v, r_l=self.r_l)
        super().setup(domain, state_fields, space=self.Vtheta)


class PotentialEnergy(ThermodynamicDiagnostic):
    """The moist compressible potential energy density."""
    name = "PotentialEnergy"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        x = SpatialCoordinate(domain.mesh)
        self._setup_thermodynamics(domain, state_fields)
        z = Function(self.rho_averaged.function_space())
        z.interpolate(dot(x, domain.k))
        self.expr = self.rho_averaged * (1 + self.r_t) * self.parameters.g * z
        super().setup(domain, state_fields, space=domain.spaces("DG"))


# TODO: this needs consolidating with energy diagnostics
class ThermodynamicKineticEnergy(ThermodynamicDiagnostic):
    """The moist compressible kinetic energy density."""
    name = "ThermodynamicKineticEnergy"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.equations = equations
        self.parameters = equations.parameters
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['rho', 'u']
            if equations.active_tracers is not None:
                for active_tracer in equations.active_tracers:
                    if active_tracer.chemical == 'H2O':
                        required_fields.append(active_tracer.name)
        else:
            raise NotImplementedError(f'Thermodynamic K.E. not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=tuple(required_fields))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields('u')
        self._setup_thermodynamics(domain, state_fields)
        self.expr = 0.5 * self.rho_averaged * (1 + self.r_t) * dot(u, u)
        super().setup(domain, state_fields, space=domain.spaces("DG"))


class Dewpoint(ThermodynamicDiagnostic):
    """The dewpoint temperature diagnostic field."""
    name = "Dewpoint"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = tde.T_dew(self.parameters, self.p, self.r_v)
        super().setup(domain, state_fields, space=self.Vtheta)


class Temperature(ThermodynamicDiagnostic):
    """The absolute temperature diagnostic field."""
    name = "Temperature"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = self.T
        super().setup(domain, state_fields, space=self.Vtheta)


class Theta_d(ThermodynamicDiagnostic):
    """The dry potential temperature diagnostic field."""
    name = "Theta_d"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = self.theta / (1 + self.r_v * self.parameters.R_v / self.parameters.R_d)
        super().setup(domain, state_fields, space=self.Vtheta)


class RelativeHumidity(ThermodynamicDiagnostic):
    """The relative humidity diagnostic field."""
    name = "RelativeHumidity"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = tde.RH(self.parameters, self.r_v, self.T, self.p)
        super().setup(domain, state_fields, space=self.Vtheta)


class Pressure(ThermodynamicDiagnostic):
    """The pressure field computed in the 'theta' space."""
    name = "Pressure_Vt"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = self.p
        super().setup(domain, state_fields, space=self.Vtheta)


class Exner_Vt(ThermodynamicDiagnostic):
    """The Exner pressure field computed in the 'theta' space."""
    name = "Exner_Vt"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = self.exner
        super().setup(domain, state_fields, space=self.Vtheta)


# TODO: this doesn't contain the effects of moisture
# TODO: this has not been implemented for other equation sets
class HydrostaticImbalance(DiagnosticField):
    """Hydrostatic imbalance diagnostic field."""
    name = "HydrostaticImbalance"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['rho', 'theta', 'rho_bar', 'theta_bar']
            if equations.active_tracers is not None:
                for active_tracer in equations.active_tracers:
                    if active_tracer.chemical == 'H2O':
                        required_fields.append(active_tracer.name)
            self.equations = equations
            self.parameters = equations.parameters
        else:
            raise NotImplementedError(f'Hydrostatic Imbalance not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=required_fields)

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        Vu = domain.spaces("HDiv")
        rho = state_fields("rho")
        rhobar = state_fields("rho_bar")
        theta = state_fields("theta")
        thetabar = state_fields("theta_bar")
        exner = tde.exner_pressure(self.parameters, rho, theta)
        exnerbar = tde.exner_pressure(self.parameters, rhobar, thetabar)

        cp = Constant(self.parameters.cp)
        n = FacetNormal(domain.mesh)

        # TODO: not sure about this expression!
        # Gravity does not appear, and why are there reference profiles?
        F = TrialFunction(Vu)
        w = TestFunction(Vu)
        imbalance = Function(Vu)
        a = inner(w, F)*dx
        L = (- cp*div((theta-thetabar)*w)*exnerbar*dx
             + cp*jump((theta-thetabar)*w, n)*avg(exnerbar)*dS_v
             - cp*div(thetabar*w)*(exner-exnerbar)*dx
             + cp*jump(thetabar*w, n)*avg(exner-exnerbar)*dS_v)

        bcs = self.equations.bcs['u']

        imbalanceproblem = LinearVariationalProblem(a, L, imbalance, bcs=bcs)
        self.imbalance_solver = LinearVariationalSolver(imbalanceproblem)
        self.expr = dot(imbalance, domain.k)
        super().setup(domain, state_fields)

    def compute(self):
        """Compute and return the diagnostic field from the current state.
        """
        self.imbalance_solver.solve()
        super().compute()


class Precipitation(DiagnosticField):
    """
    The total precipitation falling through the domain's bottom surface.

    This is normalised by unit area, giving a result in kg / m^2.
    """
    name = "Precipitation"

    def __init__(self):
        self.solve_implemented = True
        required_fields = ('rain', 'rainfall_velocity', 'rho')
        super().__init__(method='solve', required_fields=required_fields)

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        if not hasattr(domain.spaces, "DG0"):
            DG0 = domain.spaces.create_space("DG0", "DG", 0)
        else:
            DG0 = domain.spaces("DG0")
        assert DG0.extruded, 'Cannot compute precipitation on a non-extruded mesh'
        self.space = DG0

        # Gather fields
        rain = state_fields('rain')
        rho = state_fields('rho')
        v = state_fields('rainfall_velocity')
        # Set up problem
        self.phi = TestFunction(DG0)
        flux = TrialFunction(DG0)
        self.flux = Function(DG0)  # Flux to solve for
        area = Function(DG0)  # Need to compute normalisation (area)

        eqn_lhs = self.phi * flux * dx
        area_rhs = self.phi * ds_b
        eqn_rhs = domain.dt * self.phi * (rain * dot(- v, domain.k) * rho / area) * ds_b

        # Compute area normalisation
        area_prob = LinearVariationalProblem(eqn_lhs, area_rhs, area)
        area_solver = LinearVariationalSolver(area_prob)
        area_solver.solve()

        # setup solver
        rain_prob = LinearVariationalProblem(eqn_lhs, eqn_rhs, self.flux)
        self.solver = LinearVariationalSolver(rain_prob)
        self.field = state_fields(self.name, space=DG0, dump=True, pick_up=True)
        # Initialise field to zero, if picking up this will be overridden
        self.field.assign(0.0)

    def compute(self):
        """Increment the precipitation diagnostic."""
        self.solver.solve()
        self.field.assign(self.field + self.flux)


class Vorticity(DiagnosticField):
    """Base diagnostic field class for shallow-water vorticity variables."""

    def setup(self, domain, state_fields, vorticity_type=None):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
            vorticity_type (str, optional): denotes which type of vorticity to
                be computed ('relative', 'absolute' or 'potential'). Defaults to
                None.
        """

        vorticity_types = ["relative", "absolute", "potential"]
        if vorticity_type not in vorticity_types:
            raise ValueError(f"vorticity type must be one of {vorticity_types}, not {vorticity_type}")
        space = domain.spaces("H1")

        u = state_fields("u")
        if vorticity_type in ["absolute", "potential"]:
            f = state_fields("coriolis")
        if vorticity_type == "potential":
            D = state_fields("D")

        if self.method != 'solve':
            if vorticity_type == "potential":
                self.expr = curl(u + f) / D
            elif vorticity_type == "absolute":
                self.expr = curl(u + f)
            elif vorticity_type == "relative":
                self.expr = curl(u)

        super().setup(domain, state_fields, space=space)

        # Set up problem now that self.field has been set up
        if self.method == 'solve':
            gamma = TestFunction(space)
            q = TrialFunction(space)

            if vorticity_type == "potential":
                a = q*gamma*D*dx
            else:
                a = q*gamma*dx

            L = (- inner(domain.perp(grad(gamma)), u))*dx
            if vorticity_type != "relative":
                f = state_fields("coriolis")
                L += gamma*f*dx

            problem = LinearVariationalProblem(a, L, self.field)
            self.evaluator = LinearVariationalSolver(problem, solver_parameters={"ksp_type": "cg"})


class PotentialVorticity(Vorticity):
    u"""Diagnostic field for shallow-water potential vorticity, q=(∇×(u+f))/D"""
    name = "PotentialVorticity"

    def __init__(self, space=None, method='solve'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'solve'.
        """
        self.solve_implemented = True
        super().__init__(space=space, method=method,
                         required_fields=('u', 'D', 'coriolis'))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        super().setup(domain, state_fields, vorticity_type="potential")


class AbsoluteVorticity(Vorticity):
    u"""Diagnostic field for absolute vorticity, ζ=∇×(u+f)"""
    name = "AbsoluteVorticity"

    def __init__(self, space=None, method='solve'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'solve'.
        """
        self.solve_implemented = True
        super().__init__(space=space, method=method, required_fields=('u', 'coriolis'))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        super().setup(domain, state_fields, vorticity_type="absolute")


class RelativeVorticity(Vorticity):
    u"""Diagnostic field for relative vorticity, ζ=∇×u"""
    name = "RelativeVorticity"

    def __init__(self, space=None, method='solve'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'solve'.
        """
        self.solve_implemented = True
        super().__init__(space=space, method=method, required_fields=('u',))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        super().setup(domain, state_fields, vorticity_type="relative")


class TracerDensity(DiagnosticField):
    """Diagnostic for computing the density of a tracer. This is
    computed as the product of a mixing ratio and dry density"""

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return "TracerDensity_"+self.mixing_ratio_name+'_'+self.density_name

    def __init__(self, mixing_ratio_name, density_name, space=None, method='interpolate'):
        """
        Args:
            mixing_ratio_name (str): the name of the tracer mixing ratio variable
            density_name (str): the name of the tracer density variable
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project' and
                'assign'. Defaults to 'interpolate'.
        """

        super().__init__(method=method, required_fields=(mixing_ratio_name, density_name))

        self.mixing_ratio_name = mixing_ratio_name
        self.density_name = density_name

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        m_X = state_fields(self.mixing_ratio_name)
        rho_d = state_fields(self.density_name)

        m_X_space = m_X.function_space()
        rho_d_space = rho_d.function_space()

        if domain.spaces.extruded_mesh:
            m_X_horiz = m_X_space.ufl_element().sub_elements[0]
            m_X_vert = m_X_space.ufl_element().sub_elements[1]
            rho_d_horiz = rho_d_space.ufl_element().sub_elements[0]
            rho_d_vert = rho_d_space.ufl_element().sub_elements[1]

            horiz_degree = m_X_horiz.degree() + rho_d_horiz.degree()
            vert_degree = m_X_vert.degree() + rho_d_vert.degree()

            cell = domain.mesh._base_mesh.ufl_cell().cellname()
            horiz_elt = FiniteElement('DG', cell, horiz_degree)
            vert_elt = FiniteElement('DG', cell, vert_degree)
            elt = TensorProductElement(horiz_elt, vert_elt)

        else:
            m_X_degree = m_X_space.ufl_element().degree()
            rho_d_degree = rho_d_space.ufl_element().degree()
            degree = m_X_degree + rho_d_degree

            cell = domain.mesh.ufl_cell().cellname()
            elt = FiniteElement('DG', cell, degree)

        tracer_density_space = FunctionSpace(domain.mesh, elt, name='tracer_density_space')

        self.expr = m_X*rho_d
        super().setup(domain, state_fields, space=tracer_density_space)
