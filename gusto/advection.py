from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function, LinearVariationalProblem, LinearVariationalSolver


def embedded_dg(original_apply):
    def get_apply(self, x_in, x_out):
        if hasattr(self, "Projector"):
            def new_apply(self, x_in, x_out):
                self.xdg_in.interpolate(x_in)
                original_apply(self, self.xdg_in, self.xdg_out)
                self.Projector.project()
                x_out.assign(self.x_projected)
            return new_apply(self, x_in, x_out)

        else:
            return original_apply(self, x_in, x_out)
    return get_apply


class Advection(object):

    __metaclass__ = ABCMeta

    def __init__(self, state, field, equation=None, solver_params=None):

        self.state = state
        if solver_params is None:
            self.solver_parameters = {'ksp_type':'preonly',
                                      'pc_type':'bjacobi',
                                      'sub_pc_type': 'ilu'}
        else:
            self.solver_parameters = solver_params
        self.dt = self.state.timestepping.dt

        if hasattr(equation, "Projector"):
            fs = equation.xdg_out.function_space()
            self.Projector = equation.Projector
            self.xdg_out = equation.xdg_out
            self.xdg_in = Function(fs)
            self.x_projected = equation.x_projected
        else:
            fs = field.function_space()
        self.dq = Function(fs)
        self.q1 = Function(fs)
        self.equation = equation
        if equation is not None:
            self.ubar = self.equation.ubar

    def update_solver(self):
        problem = LinearVariationalProblem(self.lhs, self.rhs, self.dq)
        self.solver = LinearVariationalSolver(problem, solver_parameters=self.solver_parameters)

    @abstractmethod
    def apply(self, x_in, x_out):
        pass


class NoAdvection(Advection):
    """
    An non-advection scheme that does nothing.
    """

    def apply(self, x_in, x_out):

        x_out.assign(x_in)


class ForwardEuler(Advection):

    def __init__(self, state, field, equation, solver_params=None):
        super(ForwardEuler, self).__init__(state, field, equation, solver_params)

        self.lhs = self.equation.mass_term(self.equation.trial)
        self.rhs = self.equation.mass_term(self.q1) - self.dt*self.equation.advection_term(self.q1)
        self.update_solver()

    def apply(self, x_in, x_out):
        self.solver.solve()
        x_out.assign(x_in + self.dt*self.dq)


class SSPRK3(Advection):

    def __init__(self, state, field, equation, solver_params=None):
        super(SSPRK3, self).__init__(state, field, equation, solver_params)

        self.lhs = self.equation.mass_term(self.equation.trial)
        self.rhs = self.equation.mass_term(self.q1) - self.dt*self.equation.advection_term(self.q1)
        self.update_solver()

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

    @embedded_dg
    def apply(self, x_in, x_out):

        for i in range(3):
            self.solve_stage(x_in, x_out, i)


class ImplicitMidpoint(Advection):

    def __init__(self, state, field, equation, solver_params=None):
        super(ImplicitMidpoint, self).__init__(state, field, equation, solver_params)
        trial = self.equation.trial
        q = self.equation.q
        self.lhs = self.equation.mass_term(trial) + 0.5*self.dt*self.equation.advection_term(trial)
        self.rhs = self.equation.mass_term(q) - 0.5*self.dt*self.equation.advection_term(q)
        self.update_solver()

    def apply(self, x_in, x_out):
        self.equation.q.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)
