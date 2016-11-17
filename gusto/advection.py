from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function, LinearVariationalProblem, LinearVariationalSolver, inner, grad


def embedded_dg(original_apply):
    """
    Decorator to add interpolation and projection steps for embedded
    DG advection.
    """
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
    """
    Base class for advection schemes.

    :arg state: :class:`.State` object.
    :arg field: field to be advected
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg solver_params: solver_parameters
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, field, equation=None, solver_params=None):

        self.state = state
        self.field = field

        if solver_params is None:
            self.solver_parameters = {'ksp_type':'preonly',
                                      'pc_type':'bjacobi',
                                      'sub_pc_type': 'ilu'}
        else:
            self.solver_parameters = solver_params
        self.dt = self.state.timestepping.dt

        # check to see if we are using an embedded DG method - is we are then
        # the projector and output function will have been set up in the
        # equation class and we can get the correct function space from
        # the output function.
        if hasattr(equation, "Projector"):
            fs = equation.xdg_out.function_space()
            self.Projector = equation.Projector
            self.xdg_out = equation.xdg_out
            self.xdg_in = Function(fs)
            self.x_projected = equation.x_projected
        else:
            fs = field.function_space()

        # setup required functions
        self.dq = Function(fs)
        self.q1 = Function(fs)

        # get ubar from the equation class if provided
        self.equation = equation
        if equation is not None:
            self.ubar = self.equation.ubar

    def update_solver(self):
        # setup solver using lhs and rhs defined in derived class

        problem = LinearVariationalProblem(self.lhs, self.rhs, self.dq)
        solver_name = self.field.name()+self.equation.__class__.__name__+self.__class__.__name__
        self.solver = LinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

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
    """
    Class to implement the forward Euler timestepping scheme:
    y_(n+1) = y_n + dt*L(y_n)
    where L is the advection operator
    """
    def __init__(self, state, field, equation, solver_params=None):
        super(ForwardEuler, self).__init__(state, field, equation, solver_params)

        self.lhs = self.equation.mass_term(self.equation.trial)
        self.rhs = -self.equation.advection_term(self.q1)
        self.update_solver()

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(x_in + self.dt*self.dq)


class SSPRK3(Advection):
    """
    Class to implement the Strongly Structure Preserving Runge Kutta 3-stage
    timestepping method:
    y^1 = y_n + L(y_n)
    y^2 = (3/4)y_n + (1/4)(y^1 + L(y^1))
    y_(n+1) = (1/3)y_n + (2/3)(y^2 + L(y^2))
    where subscripts indicate the timelevel, superscripts indicate the stage
    number and L is the advection operator.
    """
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
    """
    Class to implement the implicit midpoint timestepping method:
    y_(n+1) = y_n + 0.5dt*L(y_n + y_(n+1)) where L is the advection operator.
    """
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

class TaylorGalerkin(Advection):
    def __init__(self, state, field, equation, solver_params=None):
        super(TaylorGalerkin, self).__init__(state, field, equation, solver_params)
        print "IN TG"
        self.q2 = Function(self.q1.function_space())
        self.update_solver()

    def update_solver(self):

        # stable for eta>0.473ish
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

        dt = self.dt
        trial = self.equation.trial
        q = self.equation.q
        lhs = self.equation.mass_term(trial) - eta*dt**2*self.equation.advection_term(inner(self.ubar, grad(trial)))
        rhs1 = self.equation.mass_term(q) + mu11*dt*self.equation.advection_term(q) + nu11*dt**2*self.equation.advection_term(inner(self.ubar, grad(q)))
        rhs2 = self.equation.mass_term(q) + mu21*dt*self.equation.advection_term(q) + nu21*dt**2*self.equation.advection_term(inner(self.ubar, grad(q))) + mu22*dt*self.equation.advection_term(self.q1) + nu22*dt**2*self.equation.advection_term(inner(self.ubar, grad(self.q1)))

        q1problem = LinearVariationalProblem(lhs, rhs1, self.q1)
        self.q1solver = LinearVariationalSolver(q1problem, solver_parameters={'ksp_type':'cg'})
        q2problem = LinearVariationalProblem(lhs, rhs2, self.q2)
        self.q2solver = LinearVariationalSolver(q2problem, solver_parameters={'ksp_type':'cg'})

    def apply(self, x_in, x_out):
        self.equation.q.assign(x_in)
        print "solving q1"
        self.q1solver.solve()
        print "solving q1"
        self.q2solver.solve()
        x_out.assign(self.q2)


