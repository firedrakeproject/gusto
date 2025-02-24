"""Some diagnostic fields for the Boussinesq equations."""

from firedrake import (
    as_vector, inner, dx, div, as_matrix, TrialFunction, TestFunction,
    LinearVariationalProblem, LinearVariationalSolver, DirichletBC, Function,
    grad, dot, FacetNormal, SpatialCoordinate, avg, jump, lhs, rhs, sqrt,
    FunctionSpace, dS_v, dS_h
)
from gusto.diagnostics.diagnostics import DiagnosticField

__all__ = [
    'IncompressibleGeostrophicImbalance', 'SawyerEliassenU']


class IncompressibleGeostrophicImbalance(DiagnosticField):
    """Diagnostic for the amount of geostrophic imbalance."""
    name = "GeostrophicImbalance"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`IncompressibleEadyEquations`): the equation set
                being solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.equations = equations
        self.parameters = equations.parameters
        super().__init__(space=space, method=method, required_fields=("u", "b", "p"))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        u = state_fields("u")
        b = state_fields("b")
        p = state_fields("p")
        f = self.parameters.f
        Vu = u.function_space()

        v = TrialFunction(Vu)
        w = TestFunction(Vu)
        a = inner(w, v)*dx
        L = (div(w)*p+inner(w, as_vector([f*u[1], 0.0, b])))*dx

        bcs = self.equations.bcs['u']

        imbalance = Function(Vu)
        self.expr = imbalance[0]/f
        imbalanceproblem = LinearVariationalProblem(a, L, imbalance, bcs=bcs)
        self.imbalance_solver = LinearVariationalSolver(
            imbalanceproblem, solver_parameters={'ksp_type': 'cg'})
        super().setup(domain, state_fields)

    def compute(self):
        """Compute the diagnostic field from the current state."""
        self.imbalance_solver.solve()
        super().compute()


class SawyerEliassenU(DiagnosticField):
    """
    Velocity associated with the Sawyer-Eliassen balance equation: the
    secondary circulation associated with a stream function that ensures thermal
    wind balance.
    """
    name = "SawyerEliassenU"

    def __init__(self, equations):
        """
        Args:
            equations (:class:`IncompressibleEadyEquations`): the equation set
                being solved by the model.
        """
        space = equations.domain.spaces('HDiv')
        self.parameters = equations.parameters
        self.solve_implemented = True
        super().__init__(space=space, method='solve', required_fields=("u", "b", "p"))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field
        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        super().setup(domain, state_fields)

        u = state_fields("u")
        b = state_fields("b")
        v = inner(u, as_vector([0., 1., 0.]))

        # spaces
        V0 = domain.spaces('H1')
        Vu = domain.spaces('HDiv')

        # project b to V0
        b_v0 = Function(V0)
        btri = TrialFunction(V0)
        btes = TestFunction(V0)
        a = inner(btes, btri) * dx
        L = inner(btes, b) * dx
        projectbproblem = LinearVariationalProblem(a, L, b_v0)
        self.project_b_solver = LinearVariationalSolver(
            projectbproblem, solver_parameters={'ksp_type': 'cg'})

        # project v to V0
        v_v0 = Function(V0)
        vtri = TrialFunction(V0)
        vtes = TestFunction(V0)
        a = inner(vtes, vtri) * dx
        L = inner(vtes, v) * dx
        projectvproblem = LinearVariationalProblem(a, L, v_v0)
        self.project_v_solver = LinearVariationalSolver(
            projectvproblem, solver_parameters={'ksp_type': 'cg'})

        # stm/psi is a stream function
        stm = Function(V0)
        psi = TrialFunction(V0)
        xsi = TestFunction(V0)

        f = self.parameters.f
        H = self.parameters.H
        L = self.parameters.L
        dbdy = self.parameters.dbdy
        _, _, z = SpatialCoordinate(domain.mesh)

        bcs = [DirichletBC(V0, 0., "bottom"),
               DirichletBC(V0, 0., "top")]

        Mat = as_matrix([[b.dx(2), 0., -f*v_v0.dx(2)],
                         [0., 0., 0.],
                         [-b_v0.dx(0), 0., f**2+f*v_v0.dx(0)]])

        Equ = (
            inner(grad(xsi), dot(Mat, grad(psi)))
            - dbdy*inner(grad(xsi), as_vector([-v, 0., f*(z-H/2)]))
        )*dx

        # fourth-order terms
        if self.parameters.fourthorder:
            R = FunctionSpace(domain.mesh, "R", 0)
            eps = Function(R, val=0.0001)
            brennersigma = Function(R, val=10.0)
            n = FacetNormal(domain.mesh)
            deltax = self.parameters.deltax
            deltaz = self.parameters.deltaz

            nn = as_matrix([[sqrt(brennersigma/deltax), 0., 0.],
                            [0., 0., 0.],
                            [0., 0., sqrt(brennersigma/deltaz)]])

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
        stmproblem = LinearVariationalProblem(Au, Lu, stm, bcs=bcs)
        self.stream_function_solver = LinearVariationalSolver(
            stmproblem, solver_parameters={'ksp_type': 'cg'})

        # solve for sawyer_eliassen u
        utrial = TrialFunction(Vu)
        w = TestFunction(Vu)
        a = inner(w, utrial)*dx
        L = (w[0]*(-stm.dx(2))+w[2]*(stm.dx(0)))*dx
        ugproblem = LinearVariationalProblem(a, L, self.field)
        self.sawyer_eliassen_u_solver = LinearVariationalSolver(
            ugproblem, solver_parameters={'ksp_type': 'cg'})

    def compute(self):
        """Compute the diagnostic field from the current state."""
        self.project_b_solver.solve()
        self.project_v_solver.solve()
        self.stream_function_solver.solve()
        self.sawyer_eliassen_u_solver.solve()
