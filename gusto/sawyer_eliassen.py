from __future__ import absolute_import
from os import path
from firedrake import Function, TrialFunction, TestFunction, \
    FunctionSpace, FacetNormal, inner, dx, div, grad, dot, ds_tb, \
    DirichletBC, LinearVariationalProblem, LinearVariationalSolver, \
    Constant, Expression, as_vector, as_matrix, lhs, rhs, File


class SawyerEliassenSolver(object):

    def __init__(self, state):

        self.state = state

        # setup the solver
        self._setup_solver()

    def _setup_solver(self):
        state = self.state

        V0 = FunctionSpace(state.mesh, "CG", 1)
        V1 = state.spaces("HDiv")
        V2 = state.spaces("DG")
        n = FacetNormal(state.mesh)

        u = state.fields("u")
        b = state.fields("b")
        oldv = inner(u,as_vector([0.,1.,0.]))

        self.dumpdir = path.join("results", state.output.dirname)

        # get vgrad
        self.vgrad = Function(V1)
        self.file_vgrad = File("%s/vgrad.pvd" % self.dumpdir)
        g = TrialFunction(V1)
        wg = TestFunction(V1)
        a = inner(wg,g)*dx
        L = -div(wg)*oldv*dx + inner(wg,n)*oldv*ds_tb
        vgradproblem = LinearVariationalProblem(a, L, self.vgrad)
        self.vgradsolver = LinearVariationalSolver(vgradproblem,
                                                   solver_parameters={'ksp_type': 'cg'})

        # project b to V2
        self.oldb = Function(V2)
        btest = TestFunction(V2)
        btri = TrialFunction(V2)
        a = inner(btest, btri) * dx
        L = inner(btest, b) * dx
        projectbproblem = LinearVariationalProblem(a, L, self.oldb)
        self.projectbsolver = LinearVariationalSolver(projectbproblem,
                                                      solver_parameters={'ksp_type': 'cg'})

        # get bgrad
        self.bgrad = Function(V1)
        self.file_bgrad = File("%s/bgrad.pvd" % self.dumpdir)
        g = TrialFunction(V1)
        wg = TestFunction(V1)
        a = inner(wg,g)*dx
        L = -div(wg)*self.oldb*dx + inner(wg,n)*self.oldb*ds_tb
        bgradproblem = LinearVariationalProblem(a, L, self.bgrad)
        self.bgradsolver = LinearVariationalSolver(bgradproblem,
                                                   solver_parameters={'ksp_type': 'cg'})

        # psi is a stream function
        self.stm = Function(V0)
        self.file_stm = File("%s/stm.pvd" % self.dumpdir)
        xsi = TestFunction(V0)
        psi = TrialFunction(V0)

        f = 1.e-04
        H = state.parameters.H
        Nsq = state.parameters.Nsq
        dbdy = state.parameters.dbdy
        eady_exp = Function(V0).interpolate(Expression(("x[2]-H/2"),H=H))

        bc1 = [DirichletBC(V0, Expression("0."), x)
               for x in ["top", "bottom"]]

        Equ = (
            xsi.dx(0)*Nsq*psi.dx(0) + xsi.dx(0)*b.dx(2)*psi.dx(0)
            + xsi.dx(2)*f**2*psi.dx(2) + xsi.dx(2)*f*self.vgrad[0]*psi.dx(2)
            - xsi.dx(2)*f*self.vgrad[2]*psi.dx(0) - xsi.dx(0)*self.bgrad[0]*psi.dx(2)
            + dbdy*xsi.dx(0)*oldv - dbdy*xsi.dx(2)*f*eady_exp
        )*dx

        Au = lhs(Equ)
        Lu = rhs(Equ)
        stm = Function(V0)
        stmproblem = LinearVariationalProblem(Au, Lu, self.stm, bcs=bc1)
        self.stmsolver = LinearVariationalSolver(stmproblem,
                                                 solver_parameters={'ksp_type': 'cg'})

        # get ug
        self.ug = Function(V1)
        self.file_ug = File("%s/ug.pvd" % self.dumpdir)
        utrial = TrialFunction(V1)
        w = TestFunction(V1)
        a = inner(w,utrial)*dx
        L = (w[0]*(-stm.dx(2))+w[2]*(stm.dx(0)))*dx
        ugproblem = LinearVariationalProblem(a, L, self.ug)
        self.ugsolver = LinearVariationalSolver(ugproblem,
                                                solver_parameters={'ksp_type': 'cg'})

    def solve(self):
        self.vgradsolver.solve()
        self.projectbsolver.solve()
        self.bgradsolver.solve()
        self.stmsolver.solve()
        self.ugsolver.solve()

    def dump(self):
        self.file_ug.write(self.ug)
        self.file_stm.write(self.stm)
        self.file_vgrad.write(self.vgrad)
        self.file_bgrad.write(self.bgrad)
