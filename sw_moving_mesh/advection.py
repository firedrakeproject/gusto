from firedrake import Function, Constant, dx, dS, Mesh, LinearVariationalProblem, LinearVariationalSolver, solve, lhs, rhs

class Advection(object):

    def __init__(self, field, equation, solver_params=None):

        if solver_params is None:
            self.solver_parameters = {'ksp_type':'preonly',
                                      'pc_type':'bjacobi',
                                      'sub_pc_type': 'ilu'}
        else:
            self.solver_parameters = solver_params
        fs = field.function_space()
        self.base_mesh = fs.mesh()
        self.x = self.base_mesh.coordinates
        self.x_start = Function(self.x.function_space())
        self.dq = Function(fs)
        self.q1 = Function(fs)
        self.equation = equation
        self.ubar = self.equation.ubar

    def update_solver(self):
        problem = LinearVariationalProblem(self.lhs, self.rhs, self.dq)
        self.solver = LinearVariationalSolver(problem,
                                              solver_parameters=self.solver_parameters)


class SSPRK3(Advection):

    def __init__(self, field, equation, solver_params=None):

        super(SSPRK3, self).__init__(field, equation, solver_params)

    def move_mesh(self, dt, deltax, vexpr, v, uadv, uexpr, stage):
        if stage == 0:
            self.x_start.assign(self.x)
            self.x.assign(self.x + deltax)
        elif stage == 1:
            self.x.assign(self.x - 0.5*deltax)
        elif stage == 2:
            self.x.assign(self.x + 0.5*deltax)
        v.interpolate(vexpr)
        if uexpr is not None:
            uadv.project(uexpr)
        self.ubar.project(uadv-v)

    def solve_stage(self, x_in, x_out, stage):

        if stage == 0:
            self.q1.assign(x_in)
            self.solver.solve()
            self.q1.assign(self.dq)

        elif stage == 1:
            self.solver.solve()
            self.q1.assign(0.75*x_in + 0.25*self.dq)

        elif stage == 2:
            self.solver.solve()
            x_out.assign((1./3.)*x_in + (2./3.)*self.dq)

    def apply(self, x_in, x_out, meshes, dt, deltax, vexpr, v, uadv, uexpr=None):

        lhs_domain = {0:meshes[2], 1:meshes[1], 2:meshes[2]}
        for i in range(3):
            self.lhs = self.equation.mass_term(self.equation.trial, dx(domain=lhs_domain[i]))+Constant(0)*self.equation.mass_term(self.equation.trial, dx(domain=self.base_mesh))
            self.rhs = self.equation.mass_term(self.q1, dx(domain=meshes[0])) - dt*self.equation.advection_term(self.q1, dx(domain=self.base_mesh), dS(domain=self.base_mesh))
            self.update_solver()
            self.solve_stage(x_in, x_out, i)
            self.move_mesh(dt, deltax, vexpr, v, uadv, uexpr, i)
        # self.x.assign(self.x_start)

class ImplicitMidpoint(Advection):

    def __init__(self, field, equation, solver_params=None):

        super(ImplicitMidpoint, self).__init__(field, equation, solver_params)

    def apply(self, x_in, x_out, meshes, dt, deltax, vexpr, v, uadv, uexpr=None):
        dx_lhs = dx(domain=meshes[2])
        dx_rhs = dx(domain=meshes[0])
        dS_lhs = dS(domain=meshes[2])
        dS_rhs = dS(domain=meshes[0])

        v.interpolate(vexpr)
        if uexpr is not None:
            uadv.project(uexpr)
        self.ubar.project(uadv-v)
        q = self.equation.trial

        self.lhs = self.equation.mass_term(q, dx_lhs) + 0.5*dt*self.equation.advection_term(q, dx_lhs, dS_lhs)+ Constant(0)*self.equation.mass_term(q, dx(domain=meshes[0]))
        self.x.assign(self.x + deltax)
        v.interpolate(vexpr)
        if uexpr is not None:
            uadv.project(uexpr)
        self.ubar.project(uadv-v)
        self.rhs = self.equation.mass_term(x_in, dx_rhs) - 0.5*dt*self.equation.advection_term(x_in, dx_rhs, dS_rhs)
        self.update_solver()
        self.solver.solve()
        x_out.assign(self.dq)
        self.x.assign(self.x - deltax)
