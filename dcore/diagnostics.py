from firedrake import assemble, dot, dx, FunctionSpace, Function, sqrt, \
    TestFunction, TrialFunction, Constant, div, LinearVariationalProblem, LinearVariationalSolver, inner, cross, grad, CellNormal, op2
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
    def min(f):
        fmin = op2.Global(1, [1000], dtype=float)
        op2.par_loop(op2.Kernel("""void minify(double *a, double *b)
        {
        a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
        }""", "minify"),
                     f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
        return fmin.data[0]

    @staticmethod
    def max(f):
        fmax = op2.Global(1, [-1000], dtype=float)
        op2.par_loop(op2.Kernel("""void maxify(double *a, double *b)
        {
        a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
        }""", "maxify"),
                     f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
        return fmax.data[0]

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

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        V = FunctionSpace(mesh, "CG", 2)
        self._field = Function(V, name=self.name)
        return self._field

    def solver(self, state):
        if hasattr(self, "_solver"):
            return self._solver
        u = state.field_dict['u']
        V = self.field(state.mesh).function_space()
        gamma = TestFunction(V)
        eta = TrialFunction(V)
        outward_normals = CellNormal(state.mesh)
        gradperp = lambda psi: cross(outward_normals, grad(psi))
        a = gamma*eta*dx
        L = -inner(gradperp(gamma), u)*dx
        prob = LinearVariationalProblem(a, L, self.field(state.mesh))
        self._solver = LinearVariationalSolver(prob)
        return self._solver

    def compute(self, state):

        self.solver(state).solve()
        return self.field(state.mesh)


class PotentialVorticity(DiagnosticField):
    name = "PotentialVorticity"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        V = FunctionSpace(mesh, "CG", 2)
        self._field = Function(V, name=self.name)
        return self._field

    def solver(self, state):
        if hasattr(self, "_solver"):
            return self._solver
        u = state.field_dict['u']
        D = state.field_dict['D']
        V = self.field(state.mesh).function_space()
        gamma = TestFunction(V)
        q = TrialFunction(V)
        outward_normals = CellNormal(state.mesh)
        gradperp = lambda psi: cross(outward_normals, grad(psi))
        a = gamma*q*D*dx
        L = (-inner(gradperp(gamma), u) + gamma*state.f)*dx
        prob = LinearVariationalProblem(a, L, self.field(state.mesh))
        self._solver = LinearVariationalSolver(prob)
        return self._solver

    def compute(self, state):

        self.solver(state).solve()
        return self.field(state.mesh)
