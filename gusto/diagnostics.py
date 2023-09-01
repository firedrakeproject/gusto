"""Common diagnostic fields."""

from firedrake import op2, assemble, dot, dx, Function, sqrt, \
    TestFunction, TrialFunction, Constant, grad, inner, curl, \
    LinearVariationalProblem, LinearVariationalSolver, FacetNormal, \
    ds_b, ds_v, ds_t, dS_h, dS_v, ds, dS, div, avg, jump, \
    TensorFunctionSpace, SpatialCoordinate, as_vector, \
    Projector, Interpolator
from firedrake.assign import Assigner

from abc import ABCMeta, abstractmethod, abstractproperty
import gusto.thermodynamics as tde
from gusto.recovery import Recoverer, BoundaryMethod
from gusto.equations import CompressibleEulerEquations
from gusto.active_tracers import TracerVariableType, Phases
import numpy as np

__all__ = ["Diagnostics", "CourantNumber", "VelocityX", "VelocityZ", "VelocityY", "Gradient",
           "SphericalComponent", "MeridionalComponent", "ZonalComponent", "RadialComponent",
           "RichardsonNumber", "Energy", "KineticEnergy", "ShallowWaterKineticEnergy",
           "ShallowWaterPotentialEnergy", "ShallowWaterPotentialEnstrophy",
           "CompressibleKineticEnergy", "Exner", "Sum", "Difference", "SteadyStateError",
           "Perturbation", "Theta_e", "InternalEnergy", "PotentialEnergy",
           "ThermodynamicKineticEnergy", "Dewpoint", "Temperature", "Theta_d",
           "RelativeHumidity", "Pressure", "Exner_Vt", "HydrostaticImbalance", "Precipitation",
           "PotentialVorticity", "RelativeVorticity", "AbsoluteVorticity", "Divergence",
           "TracerDensity"]


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
        # TODO check that this is correct. Maybe move the kernel elsewhere?
        """
        Finds the global minimum DoF value of a field.

        Args:
            f (:class:`Function`): field to compute diagnostic for.
        """

        fmin = op2.Global(1, np.finfo(float).max, dtype=float, comm=f._comm)
        op2.par_loop(op2.Kernel("""
static void minify(double *a, double *b) {
    a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
        return fmin.data[0]

    @staticmethod
    def max(f):
        # TODO check that this is correct. Maybe move the kernel elsewhere?
        """
        Finds the global maximum DoF value of a field.

        Args:
            f (:class:`Function`): field to compute diagnostic for.
        """

        fmax = op2.Global(1, np.finfo(float).min, dtype=float, comm=f._comm)
        op2.par_loop(op2.Kernel("""
static void maxify(double *a, double *b) {
    a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
        return fmax.data[0]

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
                    space = domain.spaces("DG0", "DG", 0)
                self.space = space
            else:
                space = self.space

            # Add space to domain
            assert space.name is not None, \
                f'Diagnostics {self.name} is using a function space which does not have a name'
            domain.spaces(space.name, V=space)

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

        V = domain.spaces("DG0", "DG", 0)
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


# TODO: unify all component diagnostics
class VelocityX(DiagnosticField):
    """The geocentric Cartesian X component of the velocity field."""
    name = "VelocityX"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields("u")
        self.expr = u[0]
        super().setup(domain, state_fields)


class VelocityZ(DiagnosticField):
    """The geocentric Cartesian Z component of the velocity field."""
    name = "VelocityZ"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields("u")
        self.expr = u[domain.mesh.geometric_dimension() - 1]
        super().setup(domain, state_fields)


class VelocityY(DiagnosticField):
    """The geocentric Cartesian Y component of the velocity field."""
    name = "VelocityY"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields("u")
        self.expr = u[1]
        super().setup(domain, state_fields)


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


class SphericalComponent(DiagnosticField):
    """Base diagnostic for computing spherical-polar components of fields."""
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

    # TODO: these routines must be moved to somewhere more available generally
    # (e.g. initialisation tools?)
    def _spherical_polar_unit_vectors(self, domain):
        """
        Generate ufl expressions for the spherical polar unit vectors.

        Args:
            domain (:class:`Domain`): the model's domain, containing its mesh.

        Returns:
            tuple of (:class:`ufl.Expr`): the zonal, meridional and radial unit
                vectors.
        """
        x, y, z = SpatialCoordinate(domain.mesh)
        x_hat = Constant(as_vector([1.0, 0.0, 0.0]))
        y_hat = Constant(as_vector([0.0, 1.0, 0.0]))
        z_hat = Constant(as_vector([0.0, 0.0, 1.0]))
        R = sqrt(x**2 + y**2)  # distance from z axis
        r = sqrt(x**2 + y**2 + z**2)  # distance from origin

        lambda_hat = (x * y_hat - y * x_hat) / R
        phi_hat = (-x*z/R * x_hat - y*z/R * y_hat + R * z_hat) / r
        r_hat = (x * x_hat + y * y_hat + z * z_hat) / r

        return lambda_hat, phi_hat, r_hat

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
        _, phi_hat, _ = self._spherical_polar_unit_vectors(domain)
        self.expr = dot(f, phi_hat)
        super().setup(domain, state_fields)


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
        lambda_hat, _, _ = self._spherical_polar_unit_vectors(domain)
        self.expr = dot(f, lambda_hat)
        super().setup(domain, state_fields)


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
        _, _, r_hat = self._spherical_polar_unit_vectors(domain)
        self.expr = dot(f, r_hat)
        super().setup(domain, state_fields)


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
            # By default set this new field to the current value
            # This may be overwritten if picking up from a checkpoint
            field2.assign(field1)

        super().setup(domain, state_fields)

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
        self.expr = self.rho_averaged * (1 + self.r_t) * self.parameters.g * dot(x, domain.k)
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
    """The total precipitation falling through the domain's bottom surface."""
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
        space = domain.spaces("DG0", "DG", 0)
        assert space.extruded, 'Cannot compute precipitation on a non-extruded mesh'
        rain = state_fields('rain')
        rho = state_fields('rho')
        v = state_fields('rainfall_velocity')
        # Set up problem
        self.phi = TestFunction(space)
        flux = TrialFunction(space)
        n = FacetNormal(domain.mesh)
        un = 0.5 * (dot(v, n) + abs(dot(v, n)))
        self.flux = Function(space)

        a = self.phi * flux * dx
        L = self.phi * rain * un * rho * (ds_b + ds_t + ds_v)

        # setup solver
        problem = LinearVariationalProblem(a, L, self.flux)
        self.solver = LinearVariationalSolver(problem)
        self.space = space
        self.field = state_fields(self.name, space=space, dump=True, pick_up=False)
        # TODO: might we want to pick up this field? Otherwise initialise to zero
        self.field.assign(0.0)

    def compute(self):
        """Compute the diagnostic field from the current state."""
        self.solver.solve()
        self.field.assign(self.field + assemble(self.flux * self.phi * dx))


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

    name = "TracerDensity"
    
    def __init__(self, m_X, rho_d):
        """
        Args:
            m_X: the mixing ratio of the tracer
            rho_d: the dry density of the tracer
        """
        super().__init__(method='interpolate', required_fields=(m_X, rho_d))
        self.m_X = m_X
        self.rho_d = rho_d

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        m_X = state_fields(self.m_X)
        rho_d = state_fields(self.rho_d)
        space = m_X.function_space()
        self.expr = m_X*rho_d
        super().setup(domain, state_fields, space=space)