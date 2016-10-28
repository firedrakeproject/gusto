from firedrake import Function, TestFunction, TrialFunction, Constant, dx, dS, inner, grad, dot, jump, Mesh, FacetNormal, LinearVariationalProblem, LinearVariationalSolver, SpatialCoordinate, solve

class SSPRK3(object):

    def __init__(self, field, equation, dt, solver_params=None):

        if solver_params is None:
            self.solver_parameters = {'ksp_type':'preonly',
                                      'pc_type':'bjacobi',
                                      'sub_pc_type': 'ilu'}
        else:
            self.solver_parameters = solver_params

        fs = field.function_space()
        self.base_mesh = fs.mesh()
        self.x = self.base_mesh.coordinates
        self.dD = Function(fs)
        self.D1 = Function(fs)
        self.equation = equation
        self.ubar = self.equation.ubar
        self.mass = self.equation.mass_term(self.equation.trial, dx)+Constant(0)*self.equation.mass_term(self.equation.trial, dx)
        self.rhs = self.equation.mass_term(self.D1, dx) - dt*self.equation.advection_term(self.D1, dx, dS)
        self.update_solver()

    def move_mesh(self, t, dt, deltax, v, uexpr, uadv, stage):
        if stage == 0:
            self.x.dat.data[:] = self.x2.dat.data[:]
        elif stage == 1:
            self.x.dat.data[:] = self.x1.dat.data[:]
        elif stage == 2:
            self.x.dat.data[:] = self.x2.dat.data[:]
        uadv.project(uexpr)
        v.project(v)
        self.ubar.project(uadv-v)
        
    def update_solver(self):
        problem = LinearVariationalProblem(self.mass, self.rhs, self.dD)
        self.solver = LinearVariationalSolver(problem,
                                              solver_parameters=self.solver_parameters)

    def solve_stage(self, x_in, x_out, stage):

        if stage == 0:
            self.D1.assign(x_in)
            self.solver.solve()
            self.D1.assign(self.dD)

        elif stage == 1:
            self.solver.solve()
            self.D1.assign(0.75*x_in + 0.25*self.dD)

        elif stage == 2:
            self.solver.solve()
            x_out.assign((1./3.)*x_in + (2./3.)*self.dD)

    def apply(self, x_in, x_out, meshes, xs, t, dt, deltax, v, uexpr, uadv):

        lhs_domain = {0:meshes[2], 1:meshes[1], 2:meshes[2]}
        self.x1 = xs[1]
        self.x2 = xs[2]
        for i in range(3):
            self.mass = self.equation.mass_term(self.equation.trial, dx(domain=lhs_domain[i]))+Constant(0)*self.equation.mass_term(self.equation.trial, dx(domain=self.base_mesh))
            self.rhs = self.equation.mass_term(self.D1, dx(domain=meshes[0])) - dt*self.equation.advection_term(self.D1, dx(domain=self.base_mesh), dS(domain=self.base_mesh))
            self.update_solver()

            self.solve_stage(x_in, x_out, i)
            self.move_mesh(t, dt, deltax, v, uexpr, uadv, i)

