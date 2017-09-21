from firedrake import op2, assemble, dot, dx, FunctionSpace, Function, sqrt, \
    TestFunction, TrialFunction, CellNormal, Constant, cross, grad, inner, \
    LinearVariationalProblem, LinearVariationalSolver, exp
from abc import ABCMeta, abstractmethod, abstractproperty
from gusto.forcing import exner
import numpy as np


__all__ = ["Diagnostics", "CourantNumber", "VelocityX", "VelocityZ", "VelocityY", "Energy", "KineticEnergy", "CompressibleKineticEnergy", "ExnerPi", "Sum", "Difference", "SteadyStateError", "Perturbation", "PotentialVorticity", "Theta_e", "InternalEnergy"]


class Diagnostics(object):

    available_diagnostics = ["min", "max", "rms", "l2", "total"]

    def __init__(self, *fields):

        self.fields = list(fields)

    def register(self, *fields):

        fset = set(self.fields)
        for f in fields:
            if f not in fset:
                self.fields.append(f)

    @staticmethod
    def min(f):
        fmin = op2.Global(1, np.finfo(float).max, dtype=float)
        op2.par_loop(op2.Kernel("""
void minify(double *a, double *b) {
    a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
        return fmin.data[0]

    @staticmethod
    def max(f):
        fmax = op2.Global(1, np.finfo(float).min, dtype=float)
        op2.par_loop(op2.Kernel("""
void maxify(double *a, double *b) {
    a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
        return fmax.data[0]

    @staticmethod
    def rms(f):
        V = FunctionSpace(f.function_space().mesh(), "DG", 1)
        c = Function(V)
        c.assign(1)
        rms = sqrt(assemble(dot(f, f)*dx)/assemble(c*dx))
        return rms

    @staticmethod
    def l2(f):
        return sqrt(assemble(dot(f, f)*dx))

    @staticmethod
    def total(f):
        if len(f.ufl_shape) == 0:
            return assemble(f * dx)
        else:
            pass


class DiagnosticField(object, metaclass=ABCMeta):

    def __init__(self):
        self._initialised = False

    @abstractproperty
    def name(self):
        """The name of this diagnostic field"""
        pass

    def setup(self, state, space=None):
        if not self._initialised:
            self._initialised = True
            if space is None:
                space = state.spaces("DG0", state.mesh, "DG", 0)
            self.field = state.fields(self.name, space, pickup=False)

    @abstractmethod
    def compute(self, state):
        """ Compute the diagnostic field from the current state"""
        pass

    def __call__(self, state):
        return self.compute(state)


class CourantNumber(DiagnosticField):
    name = "CourantNumber"

    def setup(self, state):
        if not self._initialised:
            super(CourantNumber, self).setup(state)
            # set up area computation
            V = state.spaces("DG0")
            test = TestFunction(V)
            self.area = Function(V)
            assemble(test*dx, tensor=self.area)

    def compute(self, state):
        u = state.fields("u")
        dt = Constant(state.timestepping.dt)
        return self.field.project(sqrt(dot(u, u))/sqrt(self.area)*dt)


class VelocityX(DiagnosticField):
    name = "VelocityX"

    def setup(self, state):
        if not self._initialised:
            space = state.spaces("CG1", state.mesh, "CG", 1)
            super(VelocityX, self).setup(state, space=space)

    def compute(self, state):
        u = state.fields("u")
        uh = u[0]
        return self.field.interpolate(uh)


class VelocityZ(DiagnosticField):
    name = "VelocityZ"

    def setup(self, state):
        if not self._initialised:
            space = state.spaces("CG1", state.mesh, "CG", 1)
            super(VelocityZ, self).setup(state, space=space)

    def compute(self, state):
        u = state.fields("u")
        w = u[u.geometric_dimension() - 1]
        return self.field.interpolate(w)


class VelocityY(DiagnosticField):
    name = "VelocityY"

    def setup(self, state):
        if not self._initialised:
            space = state.spaces("CG1", state.mesh, "CG", 1)
            super(VelocityY, self).setup(state, space=space)

    def compute(self, state):
        u = state.fields("u")
        v = u[1]
        return self.field.interpolate(v)


class Energy(DiagnosticField):

    def kinetic(self, u, rho=None):
        """
        Computes 0.5*dot(u, u) with an option to multiply rho
        """
        if rho is not None:
            energy = 0.5*rho*dot(u, u)
        else:
            energy = 0.5*dot(u, u)
        return energy


class KineticEnergy(Energy):
    name = "KineticEnergy"

    def compute(self, state):
        u = state.fields("u")
        energy = self.kinetic(u)
        return self.field.interpolate(energy)


class CompressibleKineticEnergy(Energy):
    name = "CompressibleKineticEnergy"

    def compute(self, state):
        u = state.fields("u")
        rho = state.fields("rho")
        energy = self.kinetic(u, rho)
        return self.field.interpolate(energy)


class ExnerPi(DiagnosticField):

    def __init__(self, reference=False):
        super(ExnerPi, self).__init__()
        self.reference = reference
        if reference:
            self.rho_name = "rhobar"
            self.theta_name = "thetabar"
        else:
            self.rho_name = "rho"
            self.theta_name = "theta"

    @property
    def name(self):
        if self.reference:
            return "ExnerPibar"
        else:
            return "ExnerPi"

    def setup(self, state):
        if not self._initialised:
            space = state.spaces("CG1", state.mesh, "CG", 1)
            super(ExnerPi, self).setup(state, space=space)

    def compute(self, state):
        rho = state.fields(self.rho_name)
        theta = state.fields(self.theta_name)
        Pi = exner(theta, rho, state)
        return self.field.interpolate(Pi)


class Theta_e(DiagnosticField):
    name = "Theta_e"

    def setup(self, state):
        if not self._initialised:
            space = state.spaces("CG1", state.mesh, "CG", 1)
            super(Theta_e, self).setup(state, space=space)

    def compute(self, state):
        X = state.parameters
        p_0 = X.p_0
        R_v = X.R_v
        R_d = X.R_d
        cp = X.cp
        c_pl = X.c_pl
        c_pv = X.c_pv
        L_v0 = X.L_v0
        T_0 = X.T_0
        kappa = X.kappa
        theta = state.fields('theta')
        rho = state.fields('rho')
        w_v = state.fields('water_v')
        p = p_0 * (R_d * theta * rho / p_0) ** (1.0 / (1.0 - kappa))
        T = theta * (R_d * theta * rho / p_0) ** (kappa / (1.0 - kappa)) / (1.0 + w_v * R_v / R_d)
        w_c = state.fields('water_c')
        w_t = w_c + w_v

        return self.field.interpolate(T * (p / (p_0 * (1 + w_v * R_v / R_d))) ** -(R_d / (cp + c_pl * w_t)) * exp(w_v * (L_v0 - (c_pl - c_pv) * (T - T_0)) / (T * (cp + c_pl * w_t))))


class InternalEnergy(DiagnosticField):
    name = "InternalEnergy"

    def setup(self, state):
        if not self._initialised:
            space = state.spaces("CG1", state.mesh, "CG", 1)
            super(InternalEnergy, self).setup(state, space=space)

    def compute(self, state):
        X = state.parameters
        p_0 = X.p_0
        R_v = X.R_v
        R_d = X.R_d
        cv = X.cv
        c_vv = X.c_vv
        c_pl = X.c_pl
        c_pv = X.c_pv
        L_v0 = X.L_v0
        T_0 = X.T_0
        kappa = X.kappa

        theta = state.fields('theta')
        rho = state.fields('rho')
        w_v = state.fields('water_v')
        w_c = state.fields('water_c')
        T = theta * (R_d * theta * rho / p_0) ** (kappa / (1.0 - kappa)) / (1.0 + w_v * R_v / R_d)

        return self.field.interpolate(rho * (cv * T + c_vv * w_v * T + c_pv * w_c * T - (L_v0 - (c_pl - c_pv) * (T - T_0)) * w_c))


class Sum(DiagnosticField):

    def __init__(self, field1, field2):
        super(Sum, self).__init__()
        self.field1 = field1
        self.field2 = field2

    @property
    def name(self):
        return self.field1+"_plus_"+self.field2

    def setup(self, state):
        if not self._initialised:
            space = state.fields(self.field1).function_space()
            super(Sum, self).setup(state, space=space)
            field_names = [f.name() for f in state.fields]
            if self.field1 not in field_names:
                raise RuntimeError("Field called %s does not exist" % self.field1)
            if self.field2 not in field_names:
                raise RuntimeError("Field called %s does not exist" % self.field2)

    def compute(self, state):
        field1 = state.fields(self.field1)
        field2 = state.fields(self.field2)
        return self.field.assign(field1 + field2)


class Difference(DiagnosticField):

    def __init__(self, field1, field2):
        super(Difference, self).__init__()
        self.field1 = field1
        self.field2 = field2

    @property
    def name(self):
        return self.field1+"_minus_"+self.field2

    def setup(self, state):
        if not self._initialised:
            space = state.fields(self.field1).function_space()
            super(Difference, self).setup(state, space=space)
            field_names = [f.name() for f in state.fields]
            if self.field1 not in field_names:
                raise RuntimeError("Field called %s does not exist" % self.field1)
            if self.field2 not in field_names:
                raise RuntimeError("Field called %s does not exist" % self.field2)

    def compute(self, state):
        field1 = state.fields(self.field1)
        field2 = state.fields(self.field2)
        return self.field.assign(field1 - field2)


class SteadyStateError(Difference):

    def __init__(self, state, name):
        DiagnosticField.__init__(self)
        self.field1 = name
        self.field2 = name+'_init'
        field1 = state.fields(name)
        field2 = state.fields(self.field2, field1.function_space())
        field2.assign(field1)

    @property
    def name(self):
        return self.field1+"_error"


class Perturbation(Difference):

    def __init__(self, name):
        DiagnosticField.__init__(self)
        self.field1 = name
        self.field2 = name+'bar'

    @property
    def name(self):
        return self.field1+"_perturbation"


class PotentialVorticity(DiagnosticField):
    """Diagnostic field for potential vorticity."""
    name = "potential_vorticity"

    def setup(self, state):
        """Solver for potential vorticity. Solves
        a weighted mass system to generate the
        potential vorticity from known velocity and
        depth fields.

        :arg state: The state containing model.
        """
        if not self._initialised:
            space = FunctionSpace(state.mesh, "CG", state.W[-1].ufl_element().degree() + 1)
            super(PotentialVorticity, self).setup(state, space=space)
            u = state.fields("u")
            D = state.fields("D")
            gamma = TestFunction(space)
            q = TrialFunction(space)
            f = state.fields("coriolis")

            cell_normals = CellNormal(state.mesh)
            gradperp = lambda psi: cross(cell_normals, grad(psi))

            a = q*gamma*D*dx
            L = (gamma*f - inner(gradperp(gamma), u))*dx
            pv_problem = LinearVariationalProblem(a, L, self.field, constant_jacobian=False)
            self.solver = LinearVariationalSolver(pv_problem, solver_parameters={"ksp_type": "cg"})

    def compute(self, state):
        """Computes the potential vorticity by solving
        the weighted mass system.
        """
        self.solver.solve()
        return self.field
