"""Common diagnostic fields."""

from firedrake import op2, assemble, dot, dx, FunctionSpace, Function, sqrt, \
    TestFunction, TrialFunction, Constant, grad, inner, \
    LinearVariationalProblem, LinearVariationalSolver, FacetNormal, \
    ds, ds_b, ds_v, ds_t, dS_v, div, avg, jump, DirichletBC, \
    TensorFunctionSpace, SpatialCoordinate, VectorFunctionSpace, as_vector

from abc import ABCMeta, abstractmethod, abstractproperty
from gusto import thermodynamics
from gusto.recovery import Recoverer, BoundaryMethod
import numpy as np

__all__ = ["Diagnostics", "CourantNumber", "VelocityX", "VelocityZ", "VelocityY", "Gradient",
           "SphericalComponent", "MeridionalComponent", "ZonalComponent", "RadialComponent",
           "RichardsonNumber", "Energy", "KineticEnergy", "ShallowWaterKineticEnergy",
           "ShallowWaterPotentialEnergy", "ShallowWaterPotentialEnstrophy",
           "CompressibleKineticEnergy", "Exner", "Sum", "Difference", "SteadyStateError",
           "Perturbation", "Theta_e", "InternalEnergy", "PotentialEnergy",
           "ThermodynamicKineticEnergy", "Dewpoint", "Temperature", "Theta_d",
           "RelativeHumidity", "Pressure", "Exner_Vt", "HydrostaticImbalance", "Precipitation",
           "PotentialVorticity", "RelativeVorticity", "AbsoluteVorticity", "Divergence"]


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

        fmin = op2.Global(1, np.finfo(float).max, dtype=float)
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

        fmax = op2.Global(1, np.finfo(float).min, dtype=float)
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
    def __init__(self, required_fields=()):
        """
        Args:
            required_fields (tuple, optional): tuple of names of the fields that
                are required for the computation of this diagnostic field.
                Defaults to ().
        """

        self._initialised = False
        self.required_fields = required_fields

    @abstractproperty
    def name(self):
        """The name of this diagnostic field"""
        pass

    def setup(self, state, space=None):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
            space (:class:`FunctionSpace`, optional): the function space for the
                diagnostic field to be computed in. Defaults to None, in which
                case the space will be DG0.
        """
        if not self._initialised:
            if space is None:
                space = state.spaces("DG0", "DG", 0)
            self.field = state.fields(self.name, space, pickup=False)
            self._initialised = True

    @abstractmethod
    def compute(self, state):
        """
        Compute the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.
        """
        pass

    def __call__(self, state):
        """
        Compute the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.
        """
        return self.compute(state)


class CourantNumber(DiagnosticField):
    """Dimensionless Courant number diagnostic field."""
    name = "CourantNumber"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            super(CourantNumber, self).setup(state)
            # set up area computation
            V = state.spaces("DG0")
            test = TestFunction(V)
            self.area = Function(V)
            assemble(test*dx, tensor=self.area)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        u = state.fields("u")
        dt = Constant(state.dt)
        return self.field.project(sqrt(dot(u, u))/sqrt(self.area)*dt)


class VelocityX(DiagnosticField):
    """The geocentric Cartesian X component of the velocity field."""
    name = "VelocityX"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            space = state.spaces("CG1", "CG", 1)
            super(VelocityX, self).setup(state, space=space)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        u = state.fields("u")
        uh = u[0]
        return self.field.interpolate(uh)


class VelocityZ(DiagnosticField):
    """The geocentric Cartesian Z component of the velocity field."""
    name = "VelocityZ"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            space = state.spaces("CG1", "CG", 1)
            super(VelocityZ, self).setup(state, space=space)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        u = state.fields("u")
        w = u[u.geometric_dimension() - 1]
        return self.field.interpolate(w)


class VelocityY(DiagnosticField):
    """The geocentric Cartesian Y component of the velocity field."""
    name = "VelocityY"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            space = state.spaces("CG1", "CG", 1)
            super(VelocityY, self).setup(state, space=space)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        u = state.fields("u")
        v = u[1]
        return self.field.interpolate(v)


class Gradient(DiagnosticField):
    """Diagnostic for computing the gradient of fields."""
    def __init__(self, name):
        """
        Args:
            name (str): name of the field to compute the gradient of.
        """
        super().__init__()
        self.fname = name

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_gradient"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            mesh_dim = state.mesh.geometric_dimension()
            try:
                field_dim = state.fields(self.fname).ufl_shape[0]
            except IndexError:
                field_dim = 1
            shape = (mesh_dim, ) * field_dim
            space = TensorFunctionSpace(state.mesh, "CG", 1, shape=shape)
            super().setup(state, space=space)

        f = state.fields(self.fname)
        test = TestFunction(space)
        trial = TrialFunction(space)
        n = FacetNormal(state.mesh)
        a = inner(test, trial)*dx
        L = -inner(div(test), f)*dx
        if space.extruded:
            L += dot(dot(test, n), f)*(ds_t + ds_b)
        prob = LinearVariationalProblem(a, L, self.field)
        self.solver = LinearVariationalSolver(prob)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        self.solver.solve()
        return self.field


class Divergence(DiagnosticField):
    """Diagnostic for computing the divergence of vector-valued fields."""
    name = "Divergence"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            space = state.spaces("DG1", "DG", 1)
            super(Divergence, self).setup(state, space=space)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        u = state.fields("u")
        return self.field.interpolate(div(u))


class SphericalComponent(DiagnosticField):
    """Base diagnostic for computing spherical-polar components of fields."""
    def __init__(self, name):
        """
        Args:
            name (str): name of the field to compute the component of.
        """
        super().__init__()
        self.fname = name

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            # check geometric dimension is 3D
            if state.mesh.geometric_dimension() != 3:
                raise ValueError('Spherical components only work when the geometric dimension is 3!')
            space = FunctionSpace(state.mesh, "CG", 1)
            super().setup(state, space=space)

        V = VectorFunctionSpace(state.mesh, "CG", 1)
        self.x, self.y, self.z = SpatialCoordinate(state.mesh)
        self.x_hat = Function(V).interpolate(Constant(as_vector([1.0, 0.0, 0.0])))
        self.y_hat = Function(V).interpolate(Constant(as_vector([0.0, 1.0, 0.0])))
        self.z_hat = Function(V).interpolate(Constant(as_vector([0.0, 0.0, 1.0])))
        self.R = sqrt(self.x**2 + self.y**2)  # distance from z axis
        self.r = sqrt(self.x**2 + self.y**2 + self.z**2)  # distance from origin
        self.f = state.fields(self.fname)
        if np.prod(self.f.ufl_shape) != 3:
            raise ValueError('Components can only be found of a vector function space in 3D.')


class MeridionalComponent(SphericalComponent):
    """The meridional component of a vector-valued field."""
    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_meridional"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        lambda_hat = (-self.x * self.z * self.x_hat / self.R
                      - self.y * self.z * self.y_hat / self.R
                      + self.R * self.z_hat) / self.r
        return self.field.project(dot(self.f, lambda_hat))


class ZonalComponent(SphericalComponent):
    """The zonal component of a vector-valued field."""
    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_zonal"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        phi_hat = (self.x * self.y_hat - self.y * self.x_hat) / self.R
        return self.field.project(dot(self.f, phi_hat))


class RadialComponent(SphericalComponent):
    """The radial component of a vector-valued field."""
    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.fname+"_radial"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        r_hat = (self.x * self.x_hat + self.y * self.y_hat + self.z * self.z_hat) / self.r
        return self.field.project(dot(self.f, r_hat))


class RichardsonNumber(DiagnosticField):
    """Dimensionless Richardson number diagnostic field."""
    name = "RichardsonNumber"

    def __init__(self, density_field, factor=1.):
        u"""
        Args:
            density_field (:class:`Function`): the density field.
            factor (float, optional): a factor to multiply the Brunt-Väisälä
                frequency by. Defaults to 1.
        """
        super().__init__(required_fields=(density_field, "u_gradient"))
        self.density_field = density_field
        self.factor = Constant(factor)

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        rho_grad = self.density_field+"_gradient"
        super().setup(state)
        self.grad_density = state.fields(rho_grad)
        self.gradu = state.fields("u_gradient")

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        denom = 0.
        z_dim = state.mesh.geometric_dimension() - 1
        u_dim = state.fields("u").ufl_shape[0]
        for i in range(u_dim-1):
            denom += self.gradu[i, z_dim]**2
        Nsq = self.factor*self.grad_density[z_dim]
        self.field.interpolate(Nsq/denom)
        return self.field


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

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        u = state.fields("u")
        energy = self.kinetic(u)
        return self.field.interpolate(energy)


class ShallowWaterKineticEnergy(Energy):
    """Diagnostic shallow-water kinetic energy density."""
    name = "ShallowWaterKineticEnergy"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        u = state.fields("u")
        D = state.fields("D")
        energy = self.kinetic(u, D)
        return self.field.interpolate(energy)


class ShallowWaterPotentialEnergy(Energy):
    """Diagnostic shallow-water potential energy density."""
    name = "ShallowWaterPotentialEnergy"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        g = state.parameters.g
        D = state.fields("D")
        energy = 0.5*g*D**2
        return self.field.interpolate(energy)


class ShallowWaterPotentialEnstrophy(DiagnosticField):
    """Diagnostic (dry) compressible kinetic energy density."""
    def __init__(self, base_field_name="PotentialVorticity"):
        """
        Args:
            base_field_name (str, optional): the base potential vorticity field
                to compute the enstrophy from. Defaults to "PotentialVorticity".
        """
        super().__init__()
        self.base_field_name = base_field_name

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        base_name = "SWPotentialEnstrophy"
        return "_from_".join((base_name, self.base_field_name))

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        if self.base_field_name == "PotentialVorticity":
            pv = state.fields("PotentialVorticity")
            D = state.fields("D")
            enstrophy = 0.5*pv**2*D
        elif self.base_field_name == "RelativeVorticity":
            zeta = state.fields("RelativeVorticity")
            D = state.fields("D")
            f = state.fields("coriolis")
            enstrophy = 0.5*(zeta + f)**2/D
        elif self.base_field_name == "AbsoluteVorticity":
            zeta_abs = state.fields("AbsoluteVorticity")
            D = state.fields("D")
            enstrophy = 0.5*(zeta_abs)**2/D
        else:
            raise ValueError("Don't know how to compute enstrophy with base_field_name=%s; base_field_name should be %s %s or %s." % (self.base_field_name, "RelativeVorticity", "AbsoluteVorticity", "PotentialVorticity"))
        return self.field.interpolate(enstrophy)


class CompressibleKineticEnergy(Energy):
    """Diagnostic (dry) compressible kinetic energy density."""
    name = "CompressibleKineticEnergy"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        u = state.fields("u")
        rho = state.fields("rho")
        energy = self.kinetic(u, rho)
        return self.field.interpolate(energy)


class Exner(DiagnosticField):
    """The diagnostic Exner pressure field."""
    def __init__(self, reference=False):
        """
        Args:
            reference (bool, optional): whether to compute the reference Exner
                pressure field or not. Defaults to False.
        """
        super(Exner, self).__init__()
        self.reference = reference
        if reference:
            self.rho_name = "rhobar"
            self.theta_name = "thetabar"
        else:
            self.rho_name = "rho"
            self.theta_name = "theta"

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        if self.reference:
            return "Exnerbar"
        else:
            return "Exner"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            space = state.spaces("CG1", "CG", 1)
            super(Exner, self).setup(state, space=space)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        rho = state.fields(self.rho_name)
        theta = state.fields(self.theta_name)
        exner = thermodynamics.exner_pressure(state.parameters, rho, theta)
        return self.field.interpolate(exner)


class Sum(DiagnosticField):
    """Base diagnostic for computing the sum of two fields."""
    def __init__(self, field1, field2):
        """
        Args:
            field1 (:class:`Function`): one field to be added.
            field2 (:class:`Function`): the other field to be added.
        """
        super().__init__(required_fields=(field1, field2))
        self.field1 = field1
        self.field2 = field2

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.field1+"_plus_"+self.field2

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            space = state.fields(self.field1).function_space()
            super(Sum, self).setup(state, space=space)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        field1 = state.fields(self.field1)
        field2 = state.fields(self.field2)
        return self.field.assign(field1 + field2)


class Difference(DiagnosticField):
    """Base diagnostic for calculating the difference between two fields."""
    def __init__(self, field1, field2):
        """
        Args:
            field1 (:class:`Function`): the field to be subtracted from.
            field2 (:class:`Function`): the field to be subtracted.
        """
        super().__init__(required_fields=(field1, field2))
        self.field1 = field1
        self.field2 = field2

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.field1+"_minus_"+self.field2

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            space = state.fields(self.field1).function_space()
            super(Difference, self).setup(state, space=space)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        field1 = state.fields(self.field1)
        field2 = state.fields(self.field2)
        return self.field.assign(field1 - field2)


class SteadyStateError(Difference):
    """Base diagnostic for computing the steady-state error in a field."""
    def __init__(self, state, name):
        """
        Args:
            state (:class:`State`): the model's state.
            name (str): name of the field to take the perturbation of.
        """
        DiagnosticField.__init__(self)
        self.field1 = name
        self.field2 = name+'_init'
        field1 = state.fields(name)
        field2 = state.fields(self.field2, field1.function_space())
        field2.assign(field1)

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.field1+"_error"


class Perturbation(Difference):
    """Base diagnostic for computing perturbations from a reference profile."""
    def __init__(self, name):
        """
        Args:
            name (str): name of the field to take the perturbation of.
        """
        self.field1 = name
        self.field2 = name+'bar'
        DiagnosticField.__init__(self, required_fields=(self.field1, self.field2))

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return self.field1+"_perturbation"


class ThermodynamicDiagnostic(DiagnosticField):
    """Base thermodynamic diagnostic field, computing many common fields."""

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            space = state.fields("theta").function_space()
            h_deg = space.ufl_element().degree()[0]
            v_deg = space.ufl_element().degree()[1]-1
            boundary_method = BoundaryMethod.extruded if (v_deg == 0 and h_deg == 0) else None
            super().setup(state, space=space)

            # now let's attach all of our fields
            self.u = state.fields("u")
            self.rho = state.fields("rho")
            self.theta = state.fields("theta")
            self.rho_averaged = Function(space)
            self.recoverer = Recoverer(self.rho, self.rho_averaged, boundary_method=boundary_method)
            try:
                self.r_v = state.fields("water_vapour")
            except NotImplementedError:
                self.r_v = Constant(0.0)
            try:
                self.r_c = state.fields("cloud_water")
            except NotImplementedError:
                self.r_c = Constant(0.0)
            try:
                self.rain = state.fields("rain")
            except NotImplementedError:
                self.rain = Constant(0.0)

            # now let's store the most common expressions
            self.exner = thermodynamics.exner_pressure(state.parameters, self.rho_averaged, self.theta)
            self.T = thermodynamics.T(state.parameters, self.theta, self.exner, r_v=self.r_v)
            self.p = thermodynamics.p(state.parameters, self.exner)
            self.r_l = self.r_c + self.rain
            self.r_t = self.r_v + self.r_c + self.rain

    def compute(self, state):
        """
        Compute thermodynamic auxiliary fields commonly used by diagnostics.

        Args:
            state (:class:`State`): the model's state.
        """
        self.recoverer.project()


class Theta_e(ThermodynamicDiagnostic):
    """The moist equivalent potential temperature diagnostic field."""
    name = "Theta_e"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.interpolate(thermodynamics.theta_e(state.parameters, self.T, self.p, self.r_v, self.r_t))


class InternalEnergy(ThermodynamicDiagnostic):
    """The moist compressible internal energy density."""
    name = "InternalEnergy"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.interpolate(thermodynamics.internal_energy(state.parameters, self.rho_averaged, self.T, r_v=self.r_v, r_l=self.r_l))


class PotentialEnergy(ThermodynamicDiagnostic):
    """The moist compressible potential energy density."""
    name = "PotentialEnergy"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        super().setup(state)
        self.x = SpatialCoordinate(state.mesh)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.interpolate(self.rho_averaged * (1 + self.r_t) * state.parameters.g * dot(self.x, state.k))


class ThermodynamicKineticEnergy(ThermodynamicDiagnostic):
    """The moist compressible kinetic energy density."""
    name = "ThermodynamicKineticEnergy"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.project(0.5 * self.rho_averaged * (1 + self.r_t) * dot(self.u, self.u))


class Dewpoint(ThermodynamicDiagnostic):
    """The dewpoint temperature diagnostic field."""
    name = "Dewpoint"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.interpolate(thermodynamics.T_dew(state.parameters, self.p, self.r_v))


class Temperature(ThermodynamicDiagnostic):
    """The absolute temperature diagnostic field."""
    name = "Temperature"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.assign(self.T)


class Theta_d(ThermodynamicDiagnostic):
    """The dry potential temperature diagnostic field."""
    name = "Theta_d"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.interpolate(self.theta / (1 + self.r_v * state.parameters.R_v / state.parameters.R_d))


class RelativeHumidity(ThermodynamicDiagnostic):
    """The relative humidity diagnostic field."""
    name = "RelativeHumidity"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.interpolate(thermodynamics.RH(state.parameters, self.r_v, self.T, self.p))


class Pressure(ThermodynamicDiagnostic):
    """The pressure field computed in the 'theta' space."""
    name = "Pressure_Vt"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.assign(self.p)


class Exner_Vt(ThermodynamicDiagnostic):
    """The Exner pressure field computed in the 'theta' space."""
    name = "Exner_Vt"

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        super().compute(state)

        return self.field.assign(self.exner)


class HydrostaticImbalance(DiagnosticField):
    """Hydrostatic imbalance diagnostic field."""
    name = "HydrostaticImbalance"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            Vu = state.spaces("HDiv")
            space = FunctionSpace(state.mesh, Vu.ufl_element()._elements[-1])
            super().setup(state, space=space)
            rho = state.fields("rho")
            rhobar = state.fields("rhobar")
            theta = state.fields("theta")
            thetabar = state.fields("thetabar")
            exner = thermodynamics.exner_pressure(state.parameters, rho, theta)
            exnerbar = thermodynamics.exner_pressure(state.parameters, rhobar, thetabar)

            cp = Constant(state.parameters.cp)
            n = FacetNormal(state.mesh)

            F = TrialFunction(space)
            w = TestFunction(space)
            a = inner(w, F)*dx
            L = (- cp*div((theta-thetabar)*w)*exnerbar*dx
                 + cp*jump((theta-thetabar)*w, n)*avg(exnerbar)*dS_v
                 - cp*div(thetabar*w)*(exner-exnerbar)*dx
                 + cp*jump(thetabar*w, n)*avg(exner-exnerbar)*dS_v)

            bcs = [DirichletBC(space, 0.0, "bottom"),
                   DirichletBC(space, 0.0, "top")]

            imbalanceproblem = LinearVariationalProblem(a, L, self.field, bcs=bcs)
            self.imbalance_solver = LinearVariationalSolver(imbalanceproblem)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        self.imbalance_solver.solve()
        return self.field[1]


class Precipitation(DiagnosticField):
    """The total precipitation falling through the domain's bottom surface."""
    name = "Precipitation"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        if not self._initialised:
            space = state.spaces("DG0", "DG", 0)
            super().setup(state, space=space)

            rain = state.fields('rain')
            rho = state.fields('rho')
            v = state.fields('rainfall_velocity')
            self.phi = TestFunction(space)
            flux = TrialFunction(space)
            n = FacetNormal(state.mesh)
            un = 0.5 * (dot(v, n) + abs(dot(v, n)))
            self.flux = Function(space)

            a = self.phi * flux * dx
            L = self.phi * rain * un * rho
            if space.extruded:
                L = L * (ds_b + ds_t + ds_v)
            else:
                L = L * ds

            # setup solver
            problem = LinearVariationalProblem(a, L, self.flux)
            self.solver = LinearVariationalSolver(problem)

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        self.solver.solve()
        self.field.assign(self.field + assemble(self.flux * self.phi * dx))
        return self.field


class Vorticity(DiagnosticField):
    """Base diagnostic field class for shallow-water vorticity variables."""

    def setup(self, state, vorticity_type=None):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
            vorticity_type (str, optional): denotes which type of vorticity to
                be computed ('relative', 'absolute' or 'potential'). Defaults to
                None.
        """
        if not self._initialised:
            vorticity_types = ["relative", "absolute", "potential"]
            if vorticity_type not in vorticity_types:
                raise ValueError("vorticity type must be one of %s, not %s" % (vorticity_types, vorticity_type))
            try:
                space = state.spaces("CG")
            except ValueError:
                dgspace = state.spaces("DG")
                cg_degree = dgspace.ufl_element().degree() + 2
                space = FunctionSpace(state.mesh, "CG", cg_degree)
            super().setup(state, space=space)
            u = state.fields("u")
            gamma = TestFunction(space)
            q = TrialFunction(space)

            if vorticity_type == "potential":
                D = state.fields("D")
                a = q*gamma*D*dx
            else:
                a = q*gamma*dx

            L = (- inner(state.perp(grad(gamma)), u))*dx
            if vorticity_type != "relative":
                f = state.fields("coriolis")
                L += gamma*f*dx

            problem = LinearVariationalProblem(a, L, self.field)
            self.solver = LinearVariationalSolver(problem, solver_parameters={"ksp_type": "cg"})

    def compute(self, state):
        """
        Compute and return the diagnostic field from the current state.

        Args:
            state (:class:`State`): the model's state.

        Returns:
            :class:`Function`: the diagnostic field.
        """
        self.solver.solve()
        return self.field


class PotentialVorticity(Vorticity):
    u"""Diagnostic field for shallow-water potential vorticity, q=(∇×(u+f))/D"""
    name = "PotentialVorticity"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        super().setup(state, vorticity_type="potential")


class AbsoluteVorticity(Vorticity):
    u"""Diagnostic field for absolute vorticity, ζ=∇×(u+f)"""
    name = "AbsoluteVorticity"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        super().setup(state, vorticity_type="absolute")


class RelativeVorticity(Vorticity):
    u"""Diagnostic field for relative vorticity, ζ=∇×u"""
    name = "RelativeVorticity"

    def setup(self, state):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            state (:class:`State`): the model's state.
        """
        super().setup(state, vorticity_type="relative")
