from firedrake import assemble, dot, dx, FunctionSpace, Function, sqrt, \
    TestFunction, Constant
from abc import ABCMeta, abstractmethod, abstractproperty


class Diagnostics(object):

    def __init__(self, *fields):

        self.fields = list(fields)

    def register(self, *fields):

        fset = set(self.fields)
        for f in fields:
            if f not in fset:
                self.fields.append(f)

    @staticmethod
    def l2(f):
        return sqrt(assemble(dot(f, f)*dx))


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
        u = getattr(state.fields, 'u')
        dt = Constant(state.timestepping.dt)
        return self.field(state.mesh).project(sqrt(dot(u, u))/sqrt(self.area(state.mesh))*dt)


class VerticalVelocity(DiagnosticField):
    name = "VerticalVelocity"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "CG", 1), name=self.name)
        return self._field

    def compute(self, state):
        u = getattr(state.fields, 'u')
        w = u[1]
        return self.field(state.mesh).interpolate(w)


class HorizontalalVelocity(DiagnosticField):
    name = "HorizontalalVelocity"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "CG", 1), name=self.name)
        return self._field

    def compute(self, state):
        u = getattr(state.fields, 'u')
        uh = u[0]
        return self.field(state.mesh).interpolate(uh)
