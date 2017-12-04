from firedrake import FunctionSpace, FacetElement, FiniteElement, \
    BrokenElement, InteriorElement, TestFunction, TrialFunction, \
    LinearVariationalProblem, LinearVariationalSolver, inner, dS, dx, \
    Function, FacetNormal, dot, grad, Jacobian, CellNormal, cross
from ufl.compound_expressions import pseudo_determinant_expr
from abc import ABCMeta, abstractmethod, abstractproperty


class Flux(object, metaclass=ABCMeta):

    def __init__(self, state):
        self.state = state
        self._setup_solvers()

    @abstractmethod
    def _setup_solvers(self):
        pass

    @abstractmethod
    def solve(self):
        pass


class MassFlux(Flux):

    def _setup_solvers(self):

        mesh = self.state.mesh
        n = FacetNormal(mesh)

        self.u = Function(self.state.spaces("HDiv"))
        self.D = Function(self.state.spaces("DG"))
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
        LFacet = v('+')*(un('+')*self.D('+') - un('-')*self.D('-'))*dS

        FacetProblem = LinearVariationalProblem(aFacet, LFacet, self.F_facet)
        self.FacetSolver = LinearVariationalSolver(FacetProblem,
                                                   solver_parameters={
                                                       'ksp_type':'preonly',
                                                       'pc_type':'bjacobi',
                                                       'sub_pc_type': 'ilu'})

        #interior equation
        w1 = TestFunction(V1_dual_interior)
        Fint = TrialFunction(V1_interior)
        self.F_interior = Function(V1_interior)
        aInterior = (inner(w1, Fint))*dx
        LInterior = inner(w1, self.u*self.D - self.F_facet)*dx
        InteriorProblem = LinearVariationalProblem(aInterior, LInterior,
                                                   self.F_interior)
        self.InteriorSolver = LinearVariationalSolver(InteriorProblem,
                                                      solver_parameters={
                                                          'pc_type':'bjacobi',
                                                          'sub_pc_type':'lu'})

        #Combine fluxes
        w_ = TestFunction(BrokenV1)
        Fb = TrialFunction(BrokenV1)
        aFull = inner(w_, Fb)*dx
        LFull = inner(w_, self.F_interior + self.F_facet)*dx
        self.F_full = Function(BrokenV1)
        self.Flux = Function(BrokenV1)
        FullProblem = LinearVariationalProblem(aFull, LFull, self.F_full)
        self.FullSolver = LinearVariationalSolver(FullProblem,
                                                  solver_parameters={
                                                      'ksp_type':'preonly',
                                                      'pc_type':'bjacobi',
                                                      'sub_pc_type': 'ilu'})

    def solve(self, stage, uin, Din):
        self.D.assign(Din)
        self.u.assign(uin)
        # print("Din: ", self.D.dat.data.min(), self.D.dat.data.max())
        # print("uin: ", self.u.dat.data.min(), self.u.dat.data.max())
        self.FacetSolver.solve()
        # print("facet: ", stage, self.F_facet.dat.data.min(), self.F_facet.dat.data.max())
        self.InteriorSolver.solve()
        # print("int: ", stage, self.F_interior.dat.data.min(), self.F_interior.dat.data.max())
        self.FullSolver.solve()
        # print("full: ", stage, self.F_full.dat.data.min(), self.F_full.dat.data.max())
        if stage == 0:
            self.Flux.assign((1./6.)*self.F_full)
        elif stage == 1:
            self.Flux += (1./6.)*self.F_full
        elif stage == 2:
            self.Flux += 2./3.*self.F_full
        # print(stage, self.Flux.dat.data.min(), self.Flux.dat.data.max())


class PVFlux(Flux):

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

    def _setup_solvers(self):

        V2 = self.state.spaces("DG")
        detJ = pseudo_determinant_expr(Jacobian(self.state.mesh))
        dx0 = dx('everywhere',
                 metadata = {'quadrature_degree': 6,
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
                                                        'ksp_type':'preonly',
                                                        'pc_type':'bjacobi',
                                                        'sub_pc_type':'ilu'})

        # setup pv solver
        V0 = FunctionSpace(self.state.mesh, "CG", 3)
        gamma = TestFunction(V0)
        q_ = TrialFunction(V0)
        self.q0 = Function(V0)
        self.u = Function(self.state.spaces("HDiv"))
        outward_normals = CellNormal(self.state.mesh)
        gradperp = lambda psi: cross(outward_normals, grad(psi))
        # gradperp = lambda psi: self.state.perp(grad(psi))
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
        dt = self.state.timestepping.dt

        self.q1 = Function(V0)
        self.q2 = Function(V0)
        BrokenV1 = FunctionSpace(self.state.mesh,
                                 BrokenElement(self.state.V1_elt))
        self.MassFlux = Function(BrokenV1)

    
        #two-step TG scheme for Q
        self.Dtwidh = 0.5*(self.Dtwid0+self.Dtwid1)/detJ
        Z0t = inner(grad(gamma), self.MassFlux)*self.q0*dx
        Z0tt = -inner(self.MassFlux, grad(gamma))*inner(self.MassFlux, grad(self.q0))/self.Dtwidh*dx0
        Z1t = inner(grad(gamma), self.MassFlux)*self.q1*dx
        Z1tt = -inner(self.MassFlux, grad(gamma))*inner(self.MassFlux, grad(self.q1))/self.Dtwidh*dx0
        Z0 = gamma*self.q0*self.Dtwid0/detJ*dx0

        PVStageLHS = (gamma*q_*self.Dtwid1/detJ + 
                      self.eta*dt**2*inner(self.MassFlux, grad(gamma))*inner(self.MassFlux, grad(q_)/self.Dtwidh))*dx0
        PVStage1RHS = Z0 + self.mu11*dt*Z0t + self.nu11*dt**2*Z0tt
        PVStage2RHS = Z0 + self.mu21*dt*Z0t + self.nu21*dt**2*Z0tt \
                      + self.mu22*dt*Z1t + self.nu22*dt**2*Z1tt

        q1problem = LinearVariationalProblem(PVStageLHS, PVStage1RHS, self.q1,
                                             constant_jacobian=False)
        self.q1_solver = LinearVariationalSolver(q1problem, 
                                            solver_parameters={'ksp_type': 'cg'})
        q2problem = LinearVariationalProblem(PVStageLHS, PVStage2RHS, self.q2,
                                             constant_jacobian=False)
        self.q2_solver = LinearVariationalSolver(q2problem, 
                                            solver_parameters={'ksp_type': 'cg'})

        self.Q = (-self.eta*dt*self.MassFlux/self.Dtwidh*inner(self.MassFlux, grad(self.q2)) 
                  +self.mu21*self.MassFlux*self.q0
                  -dt*self.nu21*self.MassFlux/self.Dtwidh*inner(self.MassFlux, grad(self.q0))
                  +self.mu22*self.MassFlux*self.q1
                  -dt*self.nu22*self.MassFlux/self.Dtwidh*inner(self.MassFlux, grad(self.q1)))

    def solve(self, u, D0, D1, MassFlux):
        dt = self.state.timestepping.dt

        self.u.assign(u)
        self.MassFlux.assign(MassFlux)

        self.Dpv.assign(D0)
        print("D0: ", self.Dpv.dat.data.min(), self.Dpv.dat.data.max())
        self.Dtwid_solver.solve()
        self.Dtwid0.assign(self.Dtwid)
        print("Dtwid0: ", self.Dtwid0.dat.data.min(), self.Dtwid0.dat.data.max())
        self.q_solver.solve()

        self.Dpv.assign(D1)
        print("D1: ", self.Dpv.dat.data.min(), self.Dpv.dat.data.max())
        self.Dtwid_solver.solve()
        self.Dtwid1.assign(self.Dtwid)
        print("Dtwid1: ", self.Dtwid1.dat.data.min(), self.Dtwid1.dat.data.max())

        self.q1_solver.solve()
        self.q2_solver.solve()
        print("q0: ", self.q0.dat.data.min(), self.q0.dat.data.max())
        print("q1: ", self.q1.dat.data.min(), self.q1.dat.data.max())
        print("q2: ", self.q2.dat.data.min(), self.q2.dat.data.max())
