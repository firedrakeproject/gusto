from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import Function, LinearVariationalProblem, \
    LinearVariationalSolver, Projector, Interpolator
from firedrake.utils import cached_property
from gusto.configuration import DEBUG
from gusto.transport_equation import EmbeddedDGAdvection
from firedrake import expression, function
from firedrake.parloops import par_loop, READ, INC
import ufl
import numpy as np


__all__ = ["NoAdvection", "ForwardEuler", "SSPRK3", "ThetaMethod"]


def embedded_dg(original_apply):
    """
    Decorator to add interpolation and projection steps for embedded
    DG advection.
    """
    def get_apply(self, x_in, x_out):
        if self.embedded_dg:
            def new_apply(self, x_in, x_out):
                if self.recovered:
                    recovered_apply(self, x_in)
                    original_apply(self, self.xdg_in, self.xdg_out)
                    recovered_project(self)
                else:
                    # try to interpolate to x_in but revert to projection
                    # if interpolation is not implemented for this
                    # function space
                    try:
                        self.xdg_in.interpolate(x_in)
                    except NotImplementedError:
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
    :arg solver_parameters: solver_parameters
    :arg limiter: :class:`.Limiter` object.
    """

    def __init__(self, state, field, equation=None, *, solver_parameters=None,
                 limiter=None):

        if equation is not None:

            self.state = state
            self.field = field
            self.equation = equation
            # get ubar from the equation class
            self.ubar = self.equation.ubar
            self.dt = self.state.timestepping.dt

            # get default solver options if none passed in
            if solver_parameters is None:
                self.solver_parameters = equation.solver_parameters
            else:
                self.solver_parameters = solver_parameters
                if state.output.log_level == DEBUG:
                    self.solver_parameters["ksp_monitor_true_residual"] = True

            self.limiter = limiter

        # check to see if we are using an embedded DG method - if we are then
        # the projector and output function will have been set up in the
        # equation class and we can get the correct function space from
        # the output function.
        if isinstance(equation, EmbeddedDGAdvection):
            # check that the field and the equation are compatible
            if equation.V0 != field.function_space():
                raise ValueError('The field to be advected is not compatible with the equation used.')
            self.embedded_dg = True
            fs = equation.space
            self.xdg_in = Function(fs)
            self.xdg_out = Function(fs)
            self.x_projected = Function(field.function_space())
            parameters = {'ksp_type': 'cg',
                          'pc_type': 'bjacobi',
                          'sub_pc_type': 'ilu'}
            self.Projector = Projector(self.xdg_out, self.x_projected,
                                       solver_parameters=parameters)
            self.recovered = equation.recovered
            if self.recovered:
                # set up the necessary functions
                self.x_in = Function(field.function_space())
                x_adv = Function(fs)
                x_rec = Function(equation.V_rec)
                x_brok = Function(equation.V_brok)

                # set up interpolators and projectors
                self.x_adv_interpolator = Interpolator(self.x_in, x_adv)  # interpolate before recovery
                self.x_rec_projector = Recoverer(x_adv, x_rec)  # recovered function
                # when the "average" method comes into firedrake master, this will be
                # self.x_rec_projector = Projector(self.x_in, equation.Vrec, method="average")
                self.x_brok_projector = Projector(x_rec, x_brok)  # function projected back
                self.xdg_interpolator = Interpolator(self.x_in + x_rec - x_brok, self.xdg_in)
                if self.limiter is not None:
                    self.x_brok_interpolator = Interpolator(self.xdg_out, x_brok)
                    self.x_out_projector = Recoverer(x_brok, self.x_projected)
                    # when the "average" method comes into firedrake master, this will be
                    # self.x_out_projector = Projector(x_brok, self.x_projected, method="average")
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

    def apply(self, x_in, x_out):
        x_out.assign(x_in)


class ExplicitAdvection(Advection):
    """
    Base class for explicit advection schemes.

    :arg state: :class:`.State` object.
    :arg field: field to be advected
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg subcycles: (optional) integer specifying number of subcycles to perform
    :arg solver_parameters: solver_parameters
    :arg limiter: :class:`.Limiter` object.
    """

    def __init__(self, state, field, equation=None, *, subcycles=None,
                 solver_parameters=None, limiter=None):
        super().__init__(state, field, equation,
                         solver_parameters=solver_parameters, limiter=limiter)

        # if user has specified a number of subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
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
    def __init__(self, state, field, equation, theta=0.5, solver_parameters=None):

        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super(ThetaMethod, self).__init__(state, field, equation,
                                          solver_parameters=solver_parameters)

        self.theta = theta

    @cached_property
    def lhs(self):
        eqn = self.equation
        trial = eqn.trial
        return eqn.mass_term(trial) + self.theta*self.dt*eqn.advection_term(self.state.h_project(trial))

    @cached_property
    def rhs(self):
        eqn = self.equation
        return eqn.mass_term(self.q1) - (1.-self.theta)*self.dt*eqn.advection_term(self.state.h_project(self.q1))

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


def recovered_apply(self, x_in):
    """
    Extra steps to the apply method for the recovered advection scheme.
    This provides an advection scheme for the lowest-degree family
    of spaces, but which has second order numerical accuracy.

    :arg x_in: the input set of prognostic fields.
    """
    self.x_in.assign(x_in)
    self.x_adv_interpolator.interpolate()
    self.x_rec_projector.project()
    self.x_brok_projector.project()
    self.xdg_interpolator.interpolate()


def recovered_project(self):
    """
    The projection steps for the recovered advection scheme,
    used for the lowest-degree sets of spaces. This returns the
    field to its original space, from the space the embedded DG
    advection happens in. This step acts as a limiter.
    """
    if self.limiter is not None:
        self.x_brok_interpolator.interpolate()
        self.x_out_projector.project()
    else:
        self.Projector.project()


class Recoverer(object):
    """
    A temporary piece of code that replicates the action of the
    Firedrake Projector object, using the "average" method.
    This can be removed when this method comes into the master branch.

    :arg v: the :class:`ufl.Expr` or
         :class:`.Function` to project.
    :arg v_out: :class:`.Function` to put the result in.
    """

    def __init__(self, v, v_out):

        if isinstance(v, expression.Expression) or not isinstance(v, (ufl.core.expr.Expr, function.Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(v))

        # Check shape values
        if v.ufl_shape != v_out.ufl_shape:
            raise RuntimeError('Shape mismatch between source %s and target function spaces %s in project' % (v.ufl_shape, v_out.ufl_shape))

        self._same_fspace = (isinstance(v, function.Function) and v.function_space() == v_out.function_space())
        self.v = v
        self.v_out = v_out
        self.V = v_out.function_space()

        # Check the number of local dofs
        if self.v_out.function_space().finat_element.space_dimension() != self.v.function_space().finat_element.space_dimension():
            raise RuntimeError("Number of local dofs for each field must be equal.")

        # NOTE: Any bcs on the function self.v should just work.
        # Loop over node extent and dof extent
        self._shapes = (self.V.finat_element.space_dimension(), np.prod(self.V.shape))
        self._average_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        vo[i][j] += v[i][j]/w[i][j];
        }}""" % self._shapes

    @cached_property
    def _weighting(self):
        """
        Generates a weight function for computing a projection via averaging.
        """
        w = Function(self.V)
        weight_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        w[i][j] += 1.0;
        }}""" % self._shapes

        par_loop(weight_kernel, ufl.dx, {"w": (w, INC)})
        return w

    def project(self):
        """
        Apply the recovery.
        """

        # Ensure that the function being populated is zeroed out
        self.v_out.dat.zero()
        par_loop(self._average_kernel, ufl.dx, {"vo": (self.v_out, INC),
                                                "w": (self._weighting, READ),
                                                "v": (self.v, READ)})
        return self.v_out
