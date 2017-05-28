from firedrake import SpatialCoordinate, TrialFunction, \
    TestFunction, Function, inner, div, as_vector, dx, \
    DirichletBC, Expression, LinearVariationalProblem, \
    LinearVariationalSolver, FunctionSpace
from gusto.diagnostics import DiagnosticField, \
    Energy, KineticEnergy, CompressibleKineticEnergy
from gusto.forcing import exner


class GeostrophicImbalance(DiagnosticField):
    name = "GeostrophicImbalance"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def set_solver(self, state):
        u = state.fields("u")
        b = state.fields("b")
        p = state.fields("p")
        f = state.parameters.f
        Vu = u.function_space()

        v = TrialFunction(Vu)
        w = TestFunction(Vu)
        a = inner(w,v)*dx
        L = (div(w)*p+inner(w,as_vector([f*u[1], 0.0, b])))*dx

        bc = ("0.", "0.", "0.")
        bcs = [DirichletBC(Vu, Expression(bc), "bottom"),
               DirichletBC(Vu, Expression(bc), "top")]

        self.imbalance = Function(Vu)
        imbalanceproblem = LinearVariationalProblem(a, L, self.imbalance, bcs=bcs)
        self.imbalance_solver = LinearVariationalSolver(imbalanceproblem,
                                                       solver_parameters={'ksp_type': 'cg'})

    def compute(self, state):
        self.set_solver(state)

        f = state.parameters.f
        self.imbalance_solver.solve()
        geostrophic_imbalance = self.imbalance[0]/f
        return self.field(state.mesh).interpolate(geostrophic_imbalance)


class KineticEnergyV(Energy):
    name = "KineticEnergyV"

    def compute(self, state):
        """
        Computes the kinetic energy of the y component
        """
        u = state.fields("u")
        energy = self.kinetic(u[1])
        return self.field(state.mesh).interpolate(energy)


class CompressibleKineticEnergyV(Energy):
    name = "CompressibleKineticEnergyV"

    def compute(self, state):
        """
        Computes the kinetic energy of the y component
        """
        u = state.fields("u")
        rho = state.fields("rho")
        energy = self.kinetic(u[1], rho)
        return self.field(state.mesh).interpolate(energy)


class EadyPotentialEnergy(Energy):
    name = "EadyPotentialEnergy"

    def compute(self, state):
        x, y, z = SpatialCoordinate(state.mesh)
        b = state.fields("b")
        bbar = state.fields("bbar")
        H = state.parameters.H
        potential = -(z-H/2)*(b-bbar)
        return self.field(state.mesh).interpolate(potential)


class CompressibleEadyPotentialEnergy(Energy):
    name = "CompressibleEadyPotentialEnergy"

    def compute(self, state):
        x, y, z = SpatialCoordinate(state.mesh)
        g = state.parameters.g
        cp = state.parameters.cp
        cv = state.parameters.cv
        Pi0 = state.parameters.Pi0

        rho = state.fields("rho")
        theta = state.fields("theta")
        Pi = exner(theta, rho, state)

        potential = rho*(g*z + cv*Pi*theta - cp*Pi0*theta)
        return self.field(state.mesh).interpolate(potential)


class EadyTotalEnergy(Energy):
    name = "EadyTotalEnergy"

    def compute(self, state):
        kinetic = KineticEnergy()
        potential = EadyPotentialEnergy()
        total = kinetic.compute(state) + potential.compute(state)
        return self.field(state.mesh).interpolate(total)


class CompressibleEadyTotalEnergy(Energy):
    name = "CompressibleEadyTotalEnergy"

    def compute(self, state):
        kinetic = CompressibleKineticEnergy()
        potential = CompressibleEadyPotentialEnergy()
        total = kinetic.compute(state) + potential.compute(state)
        return self.field(state.mesh).interpolate(total)
