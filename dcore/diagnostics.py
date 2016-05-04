from firedrake import assemble, dot, dx, FunctionSpace, Function, sqrt, \
    TestFunction, Constant
from abc import ABCMeta, abstractmethod


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

    __meteclass__ = ABCMeta

    @abstractmethod
    def compute(self, state):
        """ Compute the diagnostic field from the current state"""
        pass


class CourantNumber(DiagnosticField):

    def area(self, mesh):
        if hasattr(self, "_area"):
            return self._area
        V = FunctionSpace(mesh, "DG", 0)
        self._area = assemble(TestFunction(V)*dx)
        return self._area

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name="CourantNumber")
        return self._field

    def compute(self, state):
        u = state.field_dict['u']
        dt = Constant(state.timestepping.dt)
        return self.field(state.mesh).project(sqrt(dot(u, u))/sqrt(self.area(state.mesh))*dt)
