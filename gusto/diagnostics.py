from firedrake import op2, assemble, dot, dx, FunctionSpace, Function, sqrt, \
    TestFunction, TrialFunction, CellNormal, Constant, cross, grad, inner, \
    LinearVariationalProblem, LinearVariationalSolver
from abc import ABCMeta, abstractmethod, abstractproperty
from gusto.forcing import exner
import numpy as np


class Diagnostics(object):

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
        return assemble(f * dx)


class DiagnosticField(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        """The name of this diagnostic field"""
        pass

    @abstractmethod
    def compute(self, state):
        """ Compute the diagnostic field from the current state"""
        pass

    def __call__(self, state):
        return self.compute(state)


class CourantNumber(DiagnosticField):
    name = "CourantNumber"

    def area(self, mesh):
        if not hasattr(self, "_area"):
            V = FunctionSpace(mesh, "DG", 0)
            self.expr = TestFunction(V)*dx
            self._area = Function(V)
        assemble(self.expr, tensor=self._area)
        return self._area

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def compute(self, state):
        u = state.fields("u")
        dt = Constant(state.timestepping.dt)
        return self.field(state.mesh).project(sqrt(dot(u, u))/sqrt(self.area(state.mesh))*dt)


class VelocityX(DiagnosticField):
    name = "VelocityX"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "CG", 1), name=self.name)
        return self._field

    def compute(self, state):
        u = state.fields("u")
        uh = u[0]
        return self.field(state.mesh).interpolate(uh)


class VelocityZ(DiagnosticField):
    name = "VelocityZ"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "CG", 1), name=self.name)
        return self._field

    def compute(self, state):
        u = state.fields("u")
        w = u[u.geometric_dimension() - 1]
        return self.field(state.mesh).interpolate(w)


class VelocityY(DiagnosticField):
    name = "VelocityY"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "CG", 1), name=self.name)
        return self._field

    def compute(self, state):
        u = state.fields("u")
        v = u[1]
        return self.field(state.mesh).interpolate(v)


class Energy(DiagnosticField):

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

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
        return self.field(state.mesh).interpolate(energy)


class CompressibleKineticEnergy(Energy):
    name = "CompressibleKineticEnergy"

    def compute(self, state):
        u = state.fields("u")
        rho = state.fields("rho")
        energy = self.kinetic(u, rho)
        return self.field(state.mesh).interpolate(energy)


class ExnerPi(DiagnosticField):
    name = "ExnerPi"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "CG", 1), name=self.name)
        return self._field

    def compute(self, state):
        rho = state.fields("rho")
        theta = state.fields("theta")
        Pi = exner(theta, rho, state)
        return self.field(state.mesh).interpolate(Pi)


class ExnerPi_perturbation(ExnerPi):
    name = "ExnerPi_perturbation"

    def compute(self, state):
        rho = state.fields("rho")
        rhobar = state.fields("rhobar")
        theta = state.fields("theta")
        thetabar = state.fields("thetabar")
        Pi = exner(theta, rho, state)
        Pibar = exner(thetabar, rhobar, state)
        return self.field(state.mesh).interpolate(Pi-Pibar)


class Sum(DiagnosticField):

    def __init__(self, field1, field2):
        self.field1 = field1
        self.field2 = field2

    @property
    def name(self):
        if isinstance(self.field1, DiagnosticField):
            return self.field1.name+"_plus_"+self.field2.name
        else:
            return self.field1+"_plus_"+self.field2

    def field(self, field1):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(field1.function_space(), name=self.name)
        return self._field

    def compute(self, state):
        if isinstance(self.field1, DiagnosticField):
            field1 = self.field1.compute(state)
            field2 = self.field2.compute(state)
        else:
            field1 = state.fields(self.field1)
            field2 = state.fields(self.field2)
        return self.field(field1).assign(field1 + field2)


class Difference(DiagnosticField):

    def __init__(self, field1, field2):
        self.field1 = field1
        self.field2 = field2

    @property
    def name(self):
        if isinstance(self.field1, DiagnosticField):
            return self.field1.name+"_minus_"+self.field2.name
        else:
            return self.field1+"_minus_"+self.field2

    def field(self, field1):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(field1.function_space(), name=self.name)
        return self._field

    def compute(self, state):
        if isinstance(self.field1, DiagnosticField):
            field1 = self.field1.compute(state)
            field2 = self.field2.compute(state)
        else:
            field1 = state.fields(self.field1)
            field2 = state.fields(self.field2)
        return self.field(field1).assign(field1 - field2)


class SteadyStateError(Difference):

    def __init__(self, state, name):
        self.field1 = name
        self.field2 = name+'_init'
        field1 = state.fields(name)
        field2 = state.fields(self.field2, field1.function_space())
        field2.assign(field1)

    @property
    def name(self):
        return self.field1+"_error"


class Perturbation(Difference):

    def __init__(self, state, name):
        self.field1 = name
        self.field2 = name+'bar'

    @property
    def name(self):
        return self.field1+"_perturbation"


class PotentialVorticity(DiagnosticField):
    """Diagnostic field for potential vorticity."""
    name = "potential_vorticity"

    def field(self, space):
        """Returns the potential vorticity field.

        :arg space: The continuous finite element space.
        """
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(space, name=self.name)
        return self._field

    def solver(self, state, space):
        """Solver for potential vorticity. Solves
        a weighted mass system to generate the
        potential vorticity from known velocity and
        depth fields.

        :arg state: The state containing model.
        :arg space: The continuous finite element space.
        """
        if hasattr(self, "_solver"):
            return self._solver

        u = state.fields("u")
        D = state.fields("D")
        gamma = TestFunction(space)
        q = TrialFunction(space)
        f = state.fields("coriolis", space)

        cell_normals = CellNormal(state.mesh)
        gradperp = lambda psi: cross(cell_normals, grad(psi))

        a = q*gamma*D*dx
        L = (gamma*f - inner(gradperp(gamma), u))*dx
        pv_problem = LinearVariationalProblem(a, L, self.field(space), constant_jacobian=False)
        solver = LinearVariationalSolver(pv_problem, solver_parameters={"ksp_type": "cg"})
        self._solver = solver
        return self._solver

    def compute(self, state):
        """Computes the potential vorticity by solving
        the weighted mass system.
        """
        if hasattr(self, "_space"):
            V = self._space
        else:
            self._space = FunctionSpace(state.mesh, "CG", state.W[-1].ufl_element().degree() + 1)
            V = self._space

        self.solver(state, V).solve()
        return self.field(V)
