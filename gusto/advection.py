from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function, LinearVariationalProblem, LinearVariationalSolver
from gusto.transport_equation import LinearAdvection


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
            if isinstance(equation, LinearAdvection):
                self.solver_parameters = {'ksp_type':'cg',
                                          'pc_type':'bjacobi',
                                          'sub_pc_type': 'ilu'}
            else:
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
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
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

    def solve_stage(self, x_in, stage):

        if stage == 0:
            self.solver.solve()
            self.q1.assign(self.dq)

        elif stage == 1:
            self.solver.solve()
            self.q1.assign(0.75*x_in + 0.25*self.dq)

        elif stage == 2:
            self.solver.solve()

    @embedded_dg
    def apply(self, x_in, x_out):

        self.q1.assign(x_in)
        for i in range(3):
            self.solve_stage(x_in, i)
        x_out.assign((1./3.)*x_in + (2./3.)*self.dq)


class ThetaMethod(Advection):
    """
    Class to implement the theta timestepping method:
    y_(n+1) = y_n + dt*(theta*L(y_n) + (1-theta)*L(y_(n+1))) where L is the advection operator.
    """
    def __init__(self, state, field, equation, theta=0.5, solver_params=None):
        super(ThetaMethod, self).__init__(state, field, equation, solver_params)
        trial = self.equation.trial
        self.lhs = self.equation.mass_term(trial) + theta*self.dt*self.equation.advection_term(trial)
        self.rhs = self.equation.mass_term(self.q1) - (1.-theta)*self.dt*self.equation.advection_term(self.q1)
        self.update_solver()

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)
