from firedrake import SpatialCoordinate, TrialFunction, \
    TestFunction, Function, DirichletBC, Expression, \
    LinearVariationalProblem, LinearVariationalSolver, \
    FunctionSpace, lhs, rhs, inner, div, dx, grad, dot, \
    as_vector, as_matrix, dS_h, dS_v, Constant, avg, \
    sqrt, jump, FacetNormal
from gusto.diagnostics import DiagnosticField, Energy
from gusto.forcing import exner


class KineticEnergyY(Energy):
    name = "KineticEnergyY"

    def compute(self, state):
        """
        Computes the kinetic energy of the y component
        """
        u = state.fields("u")
        energy = self.kinetic(u[1])
        return self.field(state.mesh).interpolate(energy)


class CompressibleKineticEnergyY(Energy):
    name = "CompressibleKineticEnergyY"

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


class GeostrophicImbalance(DiagnosticField):
    name = "GeostrophicImbalance"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def setup_solver(self, state):
        u = state.fields("u")
        b = state.fields("b")
        p = state.fields("p")
        f = state.parameters.f
        Vu = u.function_space()

        v = TrialFunction(Vu)
        w = TestFunction(Vu)
        a = inner(w, v)*dx
        L = (div(w)*p+inner(w, as_vector([f*u[1], 0.0, b])))*dx

        bc = ("0.", "0.", "0.")
        bcs = [DirichletBC(Vu, Expression(bc), "bottom"),
               DirichletBC(Vu, Expression(bc), "top")]

        self.imbalance = Function(Vu)
        imbalanceproblem = LinearVariationalProblem(a, L, self.imbalance, bcs=bcs)
        self.imbalance_solver = LinearVariationalSolver(
            imbalanceproblem, solver_parameters={'ksp_type': 'cg'})

    def compute(self, state):
        try:
            getattr(self, "imbalance_solver")
        except AttributeError:
            self.setup_solver(state)
        finally:
            f = state.parameters.f
            self.imbalance_solver.solve()
            geostrophic_imbalance = self.imbalance[0]/f
            return self.field(state.mesh).interpolate(geostrophic_imbalance)


class TrueResidualV(DiagnosticField):
    name = "TrueResidualV"

    def field(self, mesh):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(FunctionSpace(mesh, "DG", 0), name=self.name)
        return self._field

    def setup_solver(self, state):
        unew, pnew, bnew = state.xn.split()
        uold, pold, bold = state.xb.split()
        ubar = 0.5*(unew+uold)
        H = state.parameters.H
        f = state.parameters.f
        dbdy = state.parameters.dbdy
        dt = state.timestepping.dt
        x, y, z = SpatialCoordinate(state.mesh)
        V = FunctionSpace(state.mesh, "DG", 0)

        wv = TestFunction(V)
        v = TrialFunction(V)
        vlhs = wv*v*dx
        vrhs = wv*((unew[1]-uold[1])/dt + ubar[0]*ubar[1].dx(0)
                   + ubar[2]*ubar[1].dx(2)
                   + f*ubar[0] + dbdy*(z-H/2))*dx
        self.vtres = Function(V)
        vtresproblem = LinearVariationalProblem(vlhs, vrhs, self.vtres)
        self.v_residual_solver = LinearVariationalSolver(
            vtresproblem, solver_parameters={'ksp_type': 'cg'})

    def compute(self, state):
        try:
            getattr(self, "v_residual_solver")
        except AttributeError:
            self.setup_solver(state)
        finally:
            self.v_residual_solver.solve()
            v_residual = self.vtres
            return self.field(state.mesh).interpolate(v_residual)


class SawyerEliassenU(DiagnosticField):
    name = "SawyerEliassenU"

    def field(self, state):
        if hasattr(self, "_field"):
            return self._field
        self._field = Function(state.spaces("HDiv"), name=self.name)
        return self._field

    def setup_solver(self, state):
        u = state.fields("u")
        b = state.fields("b")
        v = inner(u, as_vector([0., 1., 0.]))

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
        L = state.parameters.L
        dbdy = state.parameters.dbdy
        x, y, z = SpatialCoordinate(state.mesh)

        bcs = [DirichletBC(V0, Expression("0."), "bottom"),
               DirichletBC(V0, Expression("0."), "top")]

        Mat = as_matrix([[b.dx(2), 0., -f*self.v_v0.dx(2)],
                         [0., 0., 0.],
                         [-self.b_v0.dx(0), 0., f**2+f*self.v_v0.dx(0)]])

        Equ = (
            inner(grad(xsi), dot(Mat, grad(psi)))
            - dbdy*inner(grad(xsi), as_vector([-v, 0., f*(z-H/2)]))
        )*dx

        # fourth-order terms
        if state.parameters.fourthorder:
            eps = Constant(0.0001)
            brennersigma = Constant(10.0)
            n = FacetNormal(state.mesh)
            deltax = Constant(state.parameters.deltax)
            deltaz = Constant(state.parameters.deltaz)

            nn = as_matrix([[sqrt(brennersigma/Constant(deltax)), 0., 0.],
                            [0., 0., 0.],
                            [0., 0., sqrt(brennersigma/Constant(deltaz))]])

            mu = as_matrix([[1., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., H/L]])

            # anisotropic form
            Equ += eps*(
                div(dot(mu, grad(psi)))*div(dot(mu, grad(xsi)))*dx
                - (
                    avg(dot(dot(grad(grad(psi)), n), n))*jump(grad(xsi), n=n)
                    + avg(dot(dot(grad(grad(xsi)), n), n))*jump(grad(psi), n=n)
                    - jump(nn*grad(psi), n=n)*jump(nn*grad(xsi), n=n)
                )*(dS_h + dS_v)
            )

        Au = lhs(Equ)
        Lu = rhs(Equ)
        stmproblem = LinearVariationalProblem(Au, Lu, self.stm, bcs=bcs)
        self.stream_function_solver = LinearVariationalSolver(
            stmproblem, solver_parameters={'ksp_type': 'cg'})

        # solve for sawyer_eliassen u
        self.u = Function(Vu)
        utrial = TrialFunction(Vu)
        w = TestFunction(Vu)
        a = inner(w, utrial)*dx
        L = (w[0]*(-self.stm.dx(2))+w[2]*(self.stm.dx(0)))*dx
        ugproblem = LinearVariationalProblem(a, L, self.u)
        self.sawyer_eliassen_u_solver = LinearVariationalSolver(
            ugproblem, solver_parameters={'ksp_type': 'cg'})

    def compute(self, state):
        try:
            getattr(self, "sawyer_eliassen_u_solver")
        except AttributeError:
            self.setup_solver(state)
        finally:
            self.project_b_solver.solve()
            self.project_v_solver.solve()
            self.stream_function_solver.solve()
            self.sawyer_eliassen_u_solver.solve()
            sawyer_eliassen_u = self.u
            return self.field(state).project(sawyer_eliassen_u)
