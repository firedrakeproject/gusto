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


class VerticalVelocity(DiagnosticField):
    name = "VerticalVelocity"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "CG", 1), name=self.name)
        return self._field

    def compute(self, state):
        u = state.fields("u")
        w = u[1]
        return self.field(state.mesh).interpolate(w)


class HorizontalVelocity(DiagnosticField):
    name = "HorizontalVelocity"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "CG", 1), name=self.name)
        return self._field

    def compute(self, state):
        u = state.fields("u")
        uh = u[0]
        return self.field(state.mesh).interpolate(uh)


class Sum(DiagnosticField):

    def __init__(self, fieldname1, fieldname2):
        self.fieldname1 = fieldname1
        self.fieldname2 = fieldname2

    @property
    def name(self):
        return self.fieldname1+"_plus_"+self.fieldname2

    def field(self, field1):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(field1.function_space(), name=self.name)
        return self._field

    def compute(self, state):
        field1 = state.fields(self.fieldname1)
        field2 = state.fields(self.fieldname2)
        return self.field(field1).assign(field1 + field2)


class Difference(DiagnosticField):

    def __init__(self, fieldname1, fieldname2):
        self.fieldname1 = fieldname1
        self.fieldname2 = fieldname2

    @property
    def name(self):
        return self.fieldname1+"_minus_"+self.fieldname2

    def field(self, field1):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(field1.function_space(), name=self.name)
        return self._field

    def compute(self, state):
        field1 = state.fields(self.fieldname1)
        field2 = state.fields(self.fieldname2)
        return self.field(field1).assign(field1 - field2)


class SteadyStateError(Difference):

    def __init__(self, state, name):
        self.fieldname1 = name
        self.fieldname2 = name+'_init'
        field1 = state.fields(name)
        field2 = state.fields(self.fieldname2, field1.function_space())
        field2.assign(field1)

    @property
    def name(self):
        return self.fieldname1+"_error"


class Perturbation(Difference):

    def __init__(self, state, name):
        self.fieldname1 = name
        self.fieldname2 = name+'bar'

    @property
    def name(self):
        return self.fieldname1+"_perturbation"
