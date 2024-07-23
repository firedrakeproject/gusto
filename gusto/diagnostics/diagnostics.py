"""Common diagnostic fields."""


from firedrake import (assemble, dot, dx, Function, sqrt, TestFunction,
                       TrialFunction, Constant, grad, inner, FacetNormal,
                       LinearVariationalProblem, LinearVariationalSolver,
                       ds_b, ds_v, ds_t, dS_h, dS_v, ds, dS, div, avg, pi,
                       TensorFunctionSpace, SpatialCoordinate, as_vector,
                       Projector, Interpolator, FunctionSpace, FiniteElement,
                       TensorProductElement)
from firedrake.assign import Assigner
from ufl.domain import extract_unique_domain

from abc import ABCMeta, abstractmethod, abstractproperty
from gusto.core.coord_transforms import rotated_lonlatr_vectors
from gusto.core.logging import logger
from gusto.core.kernels import MinKernel, MaxKernel
import numpy as np

__all__ = ["Diagnostics", "DiagnosticField", "CourantNumber", "Gradient",
           "XComponent", "YComponent", "ZComponent", "MeridionalComponent",
           "ZonalComponent", "RadialComponent", "Energy", "KineticEnergy",
           "Sum", "Difference", "SteadyStateError", "Perturbation",
           "Divergence", "TracerDensity"]


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

        area = assemble(1*dx(domain=extract_unique_domain(f)))
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


class TracerDensity(DiagnosticField):
    """Diagnostic for computing the density of a tracer. This is
    computed as the product of a mixing ratio and dry density"""

    @property
    def name(self):
        """Gives the name of this diagnostic field. This records
        the mixing ratio and density names, in case multiple tracer
        densities are used."""
        return "TracerDensity_"+self.mixing_ratio_name+'_'+self.density_name

    def __init__(self, mixing_ratio_name, density_name, space=None, method='interpolate'):
        """
        Args:
            mixing_ratio_name (str): the name of the tracer mixing ratio variable
            density_name (str): the name of the tracer density variable
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a new space will be constructed for this diagnostic. This
                space will have enough a high enough degree to accurately compute
                the product of the mixing ratio and density.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project' and
                'assign'. Defaults to 'interpolate'.
        """
        super().__init__(space=space, method=method, required_fields=(mixing_ratio_name, density_name))

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
        self.expr = m_X*rho_d

        if self.space is None:
            # Construct a space for the diagnostic that has enough
            # degrees to accurately capture the tracer density. This
            # will be the sum of the degrees of the individual mixing ratio
            # and density function spaces.
            m_X_space = m_X.function_space()
            rho_d_space = rho_d.function_space()

            if domain.spaces.extruded_mesh:
                # Extract the base horizontal and vertical elements
                # for the mixing ratio and density.
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
            super().setup(domain, state_fields, space=tracer_density_space)

        else:
            super().setup(domain, state_fields)
