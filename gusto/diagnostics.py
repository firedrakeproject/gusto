from firedrake import assemble, dot, dx, FunctionSpace, Function, sqrt, \
    TestFunction, Constant, op2
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

    def setup(self, mesh, spaces, fields):
        if hasattr(self, "fs_name"):
            fs = spaces(self.fs_name)
        else:
            fs = spaces("DG0", mesh, "DG", 0)
        self.field = fields(self.name, fs, pickup=False)

    @abstractmethod
    def compute(self, state):
        """ Compute the diagnostic field from the current state"""
        pass

    def __call__(self, state):
        return self.compute(state)


class CourantNumber(DiagnosticField):
    name = "CourantNumber"

    def setup(self, mesh, spaces, fields):

        super(CourantNumber, self).setup(mesh, spaces, fields)
        # set up area computation
        V = spaces("DG0")
        test = TestFunction(V)
        self.area = Function(V)
        assemble(test*dx, tensor=self.area)

    def compute(self, state):
        u = state.fields("u")
        dt = Constant(state.timestepping.dt)
        return self.field.project(sqrt(dot(u, u))/sqrt(self.area)*dt)


class VelocityX(DiagnosticField):
    name = "VelocityX"

    def function_space(self, spaces, mesh):
        fs = spaces("CG1", mesh, "CG", 1)
        return fs

    def compute(self, state):
        u = state.fields("u")
        uh = u[0]
        return self.field.interpolate(uh)


class VelocityZ(DiagnosticField):
    name = "VelocityZ"

    def function_space(self, spaces, mesh):
        fs = spaces("CG1", mesh, "CG", 1)
        return fs

    def compute(self, state):
        u = state.fields("u")
        w = u[u.geometric_dimension() - 1]
        return self.field.interpolate(w)


class VelocityY(DiagnosticField):
    name = "VelocityY"

    def function_space(self, spaces, mesh):
        fs = spaces("CG1", mesh, "CG", 1)
        return fs

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
    name = "ExnerPi"

    def function_space(self, spaces, mesh):
        fs = spaces("CG1", mesh, "CG", 1)
        return fs

    def compute(self, state):
        rho = state.fields("rho")
        theta = state.fields("theta")
        Pi = exner(theta, rho, state)
        return self.field.interpolate(Pi)


class ExnerPi_perturbation(ExnerPi):
    name = "ExnerPi_perturbation"

    def compute(self, state):
        rho = state.fields("rho")
        rhobar = state.fields("rhobar")
        theta = state.fields("theta")
        thetabar = state.fields("thetabar")
        Pi = exner(theta, rho, state)
        Pibar = exner(thetabar, rhobar, state)
        return self.field.interpolate(Pi-Pibar)


class Sum(DiagnosticField):

    def __init__(self, field1, field2):
        self.field1 = field1
        self.field2 = field2

    @property
    def name(self):
        return self.field1+"_plus_"+self.field2

    def setup(self, mesh, spaces, fields):
        self.fs_name = fields(self.field1).function_space().name
        super(Sum, self).setup(mesh, spaces, fields)

    def compute(self, state):
        field1 = state.fields(self.field1)
        field2 = state.fields(self.field2)
        return self.field.assign(field1 + field2)


class Difference(DiagnosticField):

    def __init__(self, field1, field2):
        self.field1 = field1
        self.field2 = field2

    @property
    def name(self):
        return self.field1+"_minus_"+self.field2

    def setup(self, mesh, spaces, fields):
        self.fs_name = fields(self.field1).function_space().name
        super(Difference, self).setup(mesh, spaces, fields)

    def compute(self, state):
        field1 = state.fields(self.field1)
        field2 = state.fields(self.field2)
        return self.field.assign(field1 - field2)


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
