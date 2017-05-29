from firedrake import SpatialCoordinate, TrialFunction, \
    TestFunction, Function, DirichletBC, Expression, \
    LinearVariationalProblem, LinearVariationalSolver, \
    FunctionSpace, lhs, rhs, inner, div, dx, grad, dot, \
    as_vector, as_matrix
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

    def _setup_solver(self, state):
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
        self.imbalance_solver = LinearVariationalSolver(
            imbalanceproblem, solver_parameters={'ksp_type': 'cg'})

    def compute(self, state):
        f = state.parameters.f
        self.imbalance_solver.solve()
        geostrophic_imbalance = self.imbalance[0]/f
        return self.field(state.mesh).interpolate(geostrophic_imbalance)


class SawyerEliassenU(DiagnosticField):
    name = "SawyerEliassenU"

    def field(self, state):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(state.spaces("HDiv"), name=self.name)
        return self._field

    def _setup_solver(self, state):
        u = state.fields("u")
        b = state.fields("b")
        v = inner(u,as_vector([0.,1.,0.]))

        # spaces
        V0 = FunctionSpace(state.mesh, "CG", 2)
        Vu = u.function_space()

        # project b to V0
        self.b_v0 = Function(V0)
        btri = TrialFunction(V0)
        btes = TestFunction(V0)
        a = inner(btes, btri) * dx
        L = inner(btes, b) * dx
        projectbproblem = LinearVariationalProblem(a, L, self.b_v0)
        self.project_b_solver = LinearVariationalSolver(
            projectbproblem, solver_parameters={'ksp_type': 'cg'})

        # project v to V0
        self.v_v0 = Function(V0)
        vtri = TrialFunction(V0)
        vtes = TestFunction(V0)
        a = inner(vtes, vtri) * dx
        L = inner(vtes, v) * dx
        projectvproblem = LinearVariationalProblem(a, L, self.v_v0)
        self.project_v_solver = LinearVariationalSolver(
            projectvproblem, solver_parameters={'ksp_type': 'cg'})

        # stm/psi is a stream function
        self.stm = Function(V0)
        psi = TrialFunction(V0)
        xsi = TestFunction(V0)

        f = state.parameters.f
        H = state.parameters.H
        Nsq = state.parameters.Nsq
        dbdy = state.parameters.dbdy
        x, y, z = SpatialCoordinate(state.mesh)

        bcs = [DirichletBC(V0, Expression("0."), "bottom"),
               DirichletBC(V0, Expression("0."), "top")]

        self.Mat = as_matrix([[b.dx(2), 0, -f*self.v_v0.dx(2)],
                              [0, 0, 0],
                              [-self.b_v0.dx(0), 0, f**2+f*self.v_v0.dx(0)]])

        Equ = (inner(grad(xsi), dot(self.Mat, grad(psi)))
               - dbdy*inner(grad(xsi), as_vector([-v, 0, f*(z-H/2)]))
        )*dx

        Au = lhs(Equ)
        Lu = rhs(Equ)
        stmproblem = LinearVariationalProblem(Au, Lu, self.stm, bcs=bcs)
        self.stream_function_solver = LinearVariationalSolver(
            stmproblem, solver_parameters={'ksp_type': 'cg'})

        # solve for sawyer_eliassen u
        self.u = Function(Vu)
        utrial = TrialFunction(Vu)
        w = TestFunction(Vu)
        a = inner(w,utrial)*dx
        L = (w[0]*(-self.stm.dx(2))+w[2]*(self.stm.dx(0)))*dx
        ugproblem = LinearVariationalProblem(a, L, self.u)
        self.sawyer_eliassen_u_solver = LinearVariationalSolver(
            ugproblem, solver_parameters={'ksp_type': 'cg'})

    def compute(self, state):
        self.project_b_solver.solve()
        self.project_v_solver.solve()
        self.stream_function_solver.solve()
        self.sawyer_eliassen_u_solver.solve()
        sawyer_eliassen_u = self.u
        return self.field(state).project(sawyer_eliassen_u)


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
