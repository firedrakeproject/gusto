from firedrake import FunctionSpace, FacetElement, FiniteElement, \
    BrokenElement, InteriorElement, TestFunction, TrialFunction, \
    LinearVariationalProblem, LinearVariationalSolver, inner, dS, dx, \
    Function, FacetNormal, dot, grad, Jacobian, CellNormal, cross
from ufl.compound_expressions import pseudo_determinant_expr
from abc import ABCMeta, abstractmethod
from gusto.advection import SSPRK3
from gusto.transport_equation import AdvectionEquation


__all__ = ["MassFluxProjection", "MassFluxReconstruction", "PVFluxProjection", "PVFluxTaylorGalerkin"]


class Flux(object, metaclass=ABCMeta):

    def __init__(self, state):
        self.state = state

        # Function storing the variables
        self.xbar = Function(state.W)

        # create the solvers
        self._setup_solvers()

    @property
    def flux(self):
        return self.Flux

    @abstractmethod
    def _setup_solvers(self):
        pass

    def update_variables(self, xn, xbar):
        # update xbar
        self.xbar.assign(xbar)

    @abstractmethod
    def solve(self):
        pass


class MassFluxProjection(Flux):

    def _setup_solvers(self):
        self.u = Function(self.state.spaces("HDiv"))
        self.D = Function(self.state.spaces("DG"))
        self.Flux = Function(self.state.spaces("HDiv"))
        test = TestFunction(self.state.spaces("HDiv"))
        trial = TrialFunction(self.state.spaces("HDiv"))

        a = inner(test, trial)*dx
        L = inner(test, self.D*self.u)*dx
        prob = LinearVariationalProblem(a, L, self.Flux)
        self.solver = LinearVariationalSolver(prob)

    def solve(self):
        u, D = self.xbar.split()
        self.u.assign(u)
        self.D.assign(D)
        self.solver.solve()


class MassFluxReconstruction(Flux, SSPRK3):

    def __init__(self, state):
        D = state.fields("D")
        eqn = AdvectionEquation(state, D.function_space(), equation_form="continuity")
        SSPRK3.__init__(self, state, D, eqn)
        Flux.__init__(self, state)
        self.D0 = Function(D.function_space())
        self.D1 = Function(D.function_space())

    def _setup_solvers(self):

        mesh = self.state.mesh
        n = FacetNormal(mesh)

        self.u = Function(self.state.spaces("HDiv"))
        un = 0.5*(dot(self.u, n) + abs(dot(self.u, n)))

        # Spaces
        V1_elt = self.state.V1_elt
        V1_facet = FunctionSpace(mesh, FacetElement(V1_elt))
        cell = mesh.ufl_cell().cellname()
        V1_int_elt = FiniteElement("RTE", cell, 1)
        V1_dual_interior = FunctionSpace(mesh, BrokenElement(V1_int_elt))
        V1_interior = FunctionSpace(mesh, InteriorElement(V1_elt))
        BrokenV1 = FunctionSpace(mesh, BrokenElement(V1_elt))

        # Facet equation
        Tr = FunctionSpace(mesh, "HDiv Trace", 2)
        v = TestFunction(Tr)
        F_ = TrialFunction(V1_facet)
        self.F_facet = Function(V1_facet)
        aFacet = v('+')*inner(F_('+'), n('+'))*dS
        LFacet = v('+')*(un('+')*self.q1('+') - un('-')*self.q1('-'))*dS

        FacetProblem = LinearVariationalProblem(aFacet, LFacet, self.F_facet)
        self.FacetSolver = LinearVariationalSolver(FacetProblem,
                                                   solver_parameters={
                                                       'ksp_type': 'preonly',
                                                       'pc_type': 'bjacobi',
                                                       'sub_pc_type': 'ilu'})

        # interior equation
        w1 = TestFunction(V1_dual_interior)
        Fint = TrialFunction(V1_interior)
        self.F_interior = Function(V1_interior)
        aInterior = (inner(w1, Fint))*dx
        LInterior = inner(w1, self.u*self.q1 - self.F_facet)*dx
        InteriorProblem = LinearVariationalProblem(aInterior, LInterior,
                                                   self.F_interior)
        self.InteriorSolver = LinearVariationalSolver(InteriorProblem,
                                                      solver_parameters={
                                                          'pc_type': 'bjacobi',
                                                          'sub_pc_type': 'lu'})

        # Combine fluxes
        w_ = TestFunction(BrokenV1)
        Fb = TrialFunction(BrokenV1)
        aFull = inner(w_, Fb)*dx
        LFull = inner(w_, self.F_interior + self.F_facet)*dx
        self.F_full = Function(BrokenV1)
        self.Flux = Function(BrokenV1)
        FullProblem = LinearVariationalProblem(aFull, LFull, self.F_full)
        self.FullSolver = LinearVariationalSolver(FullProblem,
                                                  solver_parameters={
                                                      'ksp_type': 'preonly',
                                                      'pc_type': 'bjacobi',
                                                      'sub_pc_type': 'ilu'})

    def update_variables(self, xn, xbar):
        super().update_variables(xn, xbar)
        _, D = xn.split()
        self.D0.assign(D)
        self.q1.assign(D)
        u, _ = self.xbar.split()
        self.u.assign(u)
        SSPRK3.update_ubar(self, u)

    def solve_stage(self, x_in, stage):
        if stage == 0:
            self.Flux.assign((1./6.)*self.F_full)
        elif stage == 1:
            self.Flux += (1./6.)*self.F_full
        elif stage == 2:
            self.Flux += 2./3.*self.F_full
        super().solve_stage(x_in, stage)

    def solve(self):

        for i in range(3):
            self.solver.solve()
            self.FacetSolver.solve()
            self.InteriorSolver.solve()
            self.FullSolver.solve()
            self.solve_stage(self.D0, i)
        self.D1.assign(self.q1)


class PVFluxProjection(Flux):

    def __init__(self, state, mass_flux):
        super().__init__(state)
        self.F = mass_flux.flux

    def _setup_solvers(self):
        f = self.state.fields("coriolis")
        self.u = Function(self.state.spaces("HDiv"))
        self.D = Function(self.state.spaces("DG"))
        V0 = FunctionSpace(self.state.mesh, "CG", 3)
        test = TestFunction(V0)
        trial = TrialFunction(V0)
        self.q = Function(V0)
        a = test*self.D*trial*dx
        outward_normals = CellNormal(self.state.mesh)
        gradperp = lambda psi: cross(outward_normals, grad(psi))
        L = (-inner(gradperp(test), self.u) + test*f)*dx
        prob = LinearVariationalProblem(a, L, self.q)
        self.solver = LinearVariationalSolver(prob)

    @property
    def flux(self):
        return self.q*self.F

    def solve(self):
        u, D = self.xbar.split()
        self.u.assign(u)
        self.D.assign(D)
        self.solver.solve()


class PVFluxTaylorGalerkin(Flux):

    # TG advection constants
    eta = 0.48
    c1 = 0.5*(1 + (-1./3.+8*eta)**0.5)
    mu11 = c1
    mu12 = 0.
    mu21 = 0.5*(3-1./c1)
    mu22 = 0.5*(1./c1-1)
    nu11 = 0.5*c1**2-eta
    nu12 = 0.0
    nu21 = 0.25*(3*c1-1)-eta
    nu22 = 0.25*(1-c1)

    def __init__(self, state, mass_flux):
        self.F = mass_flux.flux
        super().__init__(state)
        Vdg = state.spaces("DG")
        self.D0 = Function(Vdg)
        self.D1 = mass_flux.D1

    def _setup_solvers(self):

        V2 = self.state.spaces("DG")
        detJ = pseudo_determinant_expr(Jacobian(self.state.mesh))
        dx0 = dx('everywhere',
                 metadata={'quadrature_degree': 6,
                           'representation': 'quadrature'})

        self.Dpv = Function(V2)
        self.Dtwid = Function(V2)
        self.Dtwid0 = Function(V2)
        self.Dtwid1 = Function(V2)
        self.Dtwidh = Function(V2)
        ptst = TestFunction(V2)
        ptri = TrialFunction(V2)
        atwid = ptst*ptri/detJ*dx0
        Ltwid = ptst*self.Dpv*dx0
        Dtwid_problem = LinearVariationalProblem(atwid, Ltwid, self.Dtwid)
        self.Dtwid_solver = LinearVariationalSolver(Dtwid_problem,
                                                    solver_parameters={
                                                        'ksp_type': 'preonly',
                                                        'pc_type': 'bjacobi',
                                                        'sub_pc_type': 'ilu'})

        # setup pv solver
        V0 = FunctionSpace(self.state.mesh, "CG", 3)
        gamma = TestFunction(V0)
        q_ = TrialFunction(V0)
        self.q0 = Function(V0)
        self.u = Function(self.state.spaces("HDiv"))
        gradperp = lambda psi: self.state.perp(grad(psi))
        f = self.state.fields("coriolis")

        aQ = gamma*q_*self.Dtwid0/detJ*dx0
        LQ = (-inner(gradperp(gamma), self.u) + gamma*f)*dx
        pv_problem = LinearVariationalProblem(aQ, LQ, self.q0,
                                              constant_jacobian=False)
        self.q_solver = LinearVariationalSolver(pv_problem,
                                                solver_parameters={
                                                    'ksp_type': 'cg',
                                                    'ksp_monitor': True})

        # setup TG advection
        self.dt = self.state.timestepping.dt

        self.q1 = Function(V0)
        self.q2 = Function(V0)

        # two-step TG scheme for Q
        self.Dtwidh = 0.5*(self.Dtwid0+self.Dtwid1)/detJ
        Z0t = inner(grad(gamma), self.F)*self.q0*dx
        Z0tt = -inner(self.F, grad(gamma))*inner(self.F, grad(self.q0))/self.Dtwidh*dx0
        Z1t = inner(grad(gamma), self.F)*self.q1*dx
        Z1tt = -inner(self.F, grad(gamma))*inner(self.F, grad(self.q1))/self.Dtwidh*dx0
        Z0 = gamma*self.q0*self.Dtwid0/detJ*dx0

        PVStageLHS = (
            gamma*q_*self.Dtwid1/detJ
            + self.eta*self.dt**2*(
                inner(self.F, grad(gamma))*inner(self.F, grad(q_)/self.Dtwidh))
        )*dx0

        PVStage1RHS = Z0 + self.mu11*self.dt*Z0t + self.nu11*self.dt**2*Z0tt
        PVStage2RHS = (
            Z0 + self.mu21*self.dt*Z0t + self.nu21*self.dt**2*Z0tt
            + self.mu22*self.dt*Z1t + self.nu22*self.dt**2*Z1tt
        )

        q1problem = LinearVariationalProblem(PVStageLHS, PVStage1RHS, self.q1,
                                             constant_jacobian=False)
        self.q1_solver = LinearVariationalSolver(q1problem,
                                                 solver_parameters={
                                                     'ksp_type': 'cg'})
        q2problem = LinearVariationalProblem(PVStageLHS, PVStage2RHS, self.q2,
                                             constant_jacobian=False)
        self.q2_solver = LinearVariationalSolver(q2problem,
                                                 solver_parameters={
                                                     'ksp_type': 'cg'})

    def update_variables(self, xn, xbar):
        u, D0 = xn.split()
        self.u.assign(u)
        self.D0.assign(D0)

    @property
    def flux(self):
        return (
            - self.eta*self.dt*self.F/self.Dtwidh*inner(self.F, grad(self.q2))
            + self.mu21*self.F*self.q0
            - self.dt*self.nu21*self.F/self.Dtwidh*inner(self.F, grad(self.q0))
            + self.mu22*self.F*self.q1
            - self.dt*self.nu22*self.F/self.Dtwidh*inner(self.F, grad(self.q1))
        )

    def solve(self):

        self.Dpv.assign(self.D0)
        self.Dtwid_solver.solve()
        self.Dtwid0.assign(self.Dtwid)

        self.q_solver.solve()

        self.Dpv.assign(self.D1)
        self.Dtwid_solver.solve()
        self.Dtwid1.assign(self.Dtwid)

        self.q1_solver.solve()
        self.q2_solver.solve()
