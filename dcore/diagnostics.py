from firedrake import assemble, dot, dx, FunctionSpace, Function, sqrt, \
    TestFunction, Constant, div
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
        u = state.field_dict['u']
        dt = Constant(state.timestepping.dt)
        return self.field(state.mesh).project(sqrt(dot(u, u))/sqrt(self.area(state.mesh))*dt)


class Divergence(DiagnosticField):
    name = "Divergence"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def compute(self, state):
        u = state.field_dict['u']
        return self.field(state.mesh).project(div(u))


class Vorticity(DiagnosticField):
    name = "Vorticity"

    def field(self, mesh, V):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(V, name=self.name)
        return self._field

    def solver(self, state):
        V = FunctionSpace(state.mesh, "CG", 2)
        u = state.field_dict['u']
        gamma = TestFunction(V)
        eta = TrialFunction(V)
        a = gamma*eta*dx
        L = -inner(curl(gamma), u)*dx
        prob = LinearVariationalProblem(a, L, self._field)
        self.solver(prob)

    def compute(self, state):

        self.solver.solve()
        return self.field(state.mesh)
