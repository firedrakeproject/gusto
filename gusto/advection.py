from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import Function, LinearVariationalProblem, \
    LinearVariationalSolver, Projector
from firedrake.utils import cached_property
from gusto.transport_equation import EmbeddedDGAdvection


__all__ = ["NoAdvection", "ForwardEuler", "SSPRK3", "ThetaMethod"]


def embedded_dg(original_apply):
    """
    Decorator to add interpolation and projection steps for embedded
    DG advection.
    """
    def get_apply(self, x_in, x_out):
        if self.embedded_dg:
            def new_apply(self, x_in, x_out):
                try:
                    self.xdg_in.interpolate(x_in)
                except:
                    self.xdg_in.project(x_in)
                original_apply(self, self.xdg_in, self.xdg_out)
                self.Projector.project()
                x_out.assign(self.x_projected)
            return new_apply(self, x_in, x_out)

        else:
            return original_apply(self, x_in, x_out)
    return get_apply


class Advection(object, metaclass=ABCMeta):
    """
    Base class for advection schemes.

    :arg state: :class:`.State` object.
    :arg field: field to be advected
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg solver_params: solver_parameters
    """

    def __init__(self, state, field, equation=None, solver_params=None, limiter=None):

        if equation is not None:

            self.state = state
            self.field = field
            self.equation = equation
            # get ubar from the equation class
            self.ubar = self.equation.ubar
            self.dt = self.state.timestepping.dt

            # get default solver options if none passed in
            if solver_params is None:
                self.solver_parameters = equation.solver_parameters
            else:
                self.solver_parameters = solver_params

            self.limiter = limiter

        # check to see if we are using an embedded DG method - if we are then
        # the projector and output function will have been set up in the
        # equation class and we can get the correct function space from
        # the output function.
        if isinstance(equation, EmbeddedDGAdvection):
            self.embedded_dg = True
            fs = equation.space
            self.xdg_in = Function(equation.space)
            self.xdg_out = Function(equation.space)
            self.x_projected = Function(field.function_space())
            parameters = {'ksp_type': 'cg',
                          'pc_type': 'bjacobi',
                          'sub_pc_type': 'ilu'}
            self.Projector = Projector(self.xdg_out, self.x_projected,
                                       solver_parameters=parameters)
            self.xdg_in = Function(fs)
        else:
            self.embedded_dg = False
            fs = field.function_space()

        # setup required functions
        self.fs = fs
        self.dq = Function(fs)
        self.q1 = Function(fs)

    @abstractproperty
    def lhs(self):
        return self.equation.mass_term(self.equation.trial)

    @abstractproperty
    def rhs(self):
        return self.equation.mass_term(self.q1) - self.dt*self.equation.advection_term(self.q1)

    def update_ubar(self, xn, xnp1, alpha):
        un = xn.split()[0]
        unp1 = xnp1.split()[0]
        self.ubar.assign(un + alpha*(unp1-un))

    @cached_property
    def solver(self):
        # setup solver using lhs and rhs defined in derived class
        problem = LinearVariationalProblem(self.lhs, self.rhs, self.dq)
        solver_name = self.field.name()+self.equation.__class__.__name__+self.__class__.__name__
        return LinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

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

    def lhs(self):
        pass

    def rhs(self):
        pass

    def update_ubar(self, xn, xnp1, alpha):
        pass

    def apply_cycle(self, x_in, x_out):
        pass

    def apply(self, x_in, x_out):
        x_out.assign(x_in)


class ExplicitAdvection(Advection):

    def __init__(self, state, field, equation=None, subcycles=None, solver_params=None):
        super().__init__(state, field, equation, solver_params)
        if subcycles is not None:
            self.dt = self.dt/subcycles
            self.ncycles = subcycles
        else:
            self.dt = self.dt
            self.ncycles = 1
        self.x = [Function(self.fs)]*(self.ncycles+1)

    @abstractmethod
    def apply_cycle(self, x_in, x_out):
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass

    @embedded_dg
    def apply(self, x_in, x_out):
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        self.x[0].assign(x_in)
        for i in range(self.ncycles):
            self.apply_cycle(self.x[i], self.x[i+1])
            self.x[i].assign(self.x[i+1])
        x_out.assign(self.x[self.ncycles-1])


class ForwardEuler(ExplicitAdvection):
    """
    Class to implement the forward Euler timestepping scheme:
    y_(n+1) = y_n + dt*L(y_n)
    where L is the advection operator
    """

    @cached_property
    def lhs(self):
        return super(ForwardEuler, self).lhs

    @cached_property
    def rhs(self):
        return super(ForwardEuler, self).rhs

    def apply_cycle(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class SSPRK3(ExplicitAdvection):
    """
    Class to implement the Strongly Structure Preserving Runge Kutta 3-stage
    timestepping method:
    y^1 = y_n + L(y_n)
    y^2 = (3/4)y_n + (1/4)(y^1 + L(y^1))
    y_(n+1) = (1/3)y_n + (2/3)(y^2 + L(y^2))
    where subscripts indicate the timelevel, superscripts indicate the stage
    number and L is the advection operator.
    """

    @cached_property
    def lhs(self):
        return super(SSPRK3, self).lhs

    @cached_property
    def rhs(self):
        return super(SSPRK3, self).rhs

    def solve_stage(self, x_in, stage):

        if stage == 0:
            self.solver.solve()
            self.q1.assign(self.dq)

        elif stage == 1:
            self.solver.solve()
            self.q1.assign(0.75*x_in + 0.25*self.dq)

        elif stage == 2:
            self.solver.solve()
            self.q1.assign((1./3.)*x_in + (2./3.)*self.dq)

        if self.limiter is not None:
            self.limiter.apply(self.q1)

    def apply_cycle(self, x_in, x_out):

        if self.limiter is not None:
            self.limiter.apply(x_in)

        self.q1.assign(x_in)
        for i in range(3):
            self.solve_stage(x_in, i)
        x_out.assign(self.q1)


class ThetaMethod(Advection):
    """
    Class to implement the theta timestepping method:
    y_(n+1) = y_n + dt*(theta*L(y_n) + (1-theta)*L(y_(n+1))) where L is the advection operator.
    """
    def __init__(self, state, field, equation, theta=0.5, solver_params=None):

        if not solver_params:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_params = {'ksp_type': 'gmres',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}

        super(ThetaMethod, self).__init__(state, field, equation, solver_params)

        self.theta = theta

    @cached_property
    def lhs(self):
        eqn = self.equation
        trial = eqn.trial
        return eqn.mass_term(trial) + self.theta*self.dt*eqn.advection_term(trial)

    @cached_property
    def rhs(self):
        eqn = self.equation
        return eqn.mass_term(self.q1) - (1.-self.theta)*self.dt*eqn.advection_term(self.q1)

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)
