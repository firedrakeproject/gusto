"""Objects to describe explicit multi-stage (Runge-Kutta) discretisations."""

import numpy as np

from firedrake import (Function, Constant, NonlinearVariationalProblem,
                       NonlinearVariationalSolver)
from firedrake.fml import replace_subject, all_terms, drop, keep
from firedrake.utils import cached_property

from gusto.core.labels import time_derivative
from gusto.core.logging import logger
from gusto.time_discretisation.time_discretisation import ExplicitTimeDiscretisation


__all__ = ["ForwardEuler", "ExplicitRungeKutta", "SSPRK3", "RK4", "Heun"]


class ExplicitRungeKutta(ExplicitTimeDiscretisation):
    """
    A class for implementing general explicit multistage (Runge-Kutta)
    methods based on its Butcher tableau.

    A Butcher tableau is formed in the following way for a s-th order explicit
    scheme:                                                                   \n

    All upper diagonal a_ij elements are zero for explicit methods.

    There are three steps to move from the current solution, y^n, to the new
    one, y^{n+1}

    For each i = 1, s  in an s stage method
    we have the intermediate solutions:                                       \n
    y_i = y^n +  dt*(a_i1*k_1 + a_i2*k_2 + ... + a_i{i-1}*k_{i-1})            \n
    We compute the gradient at the intermediate location, k_i = F(y_i)        \n

    At the last stage, compute the new solution by:
    y^{n+1} = y^n + dt*(b_1*k_1 + b_2*k_2 + .... + b_s*k_s)                   \n

    """
    # ---------------------------------------------------------------------------
    # Butcher tableau for a s-th order
    # explicit scheme:
    #  c_0 |   0   0    .    0
    #  c_1 | a_10  0    .    0
    #   .  |   .   .    .    .
    #   .  |   .   .    .    .
    #  c_s | a_s0 a_s1  .    0
    #   -------------------------
    #      |  b_1  b_2  ...  b_s
    #
    #
    # The corresponding square 'butcher_matrix' is:
    #
    #  [a_10   0   .       0  ]
    #  [a_20  a_21 .       0  ]
    #  [  .    .   .       .  ]
    #  [ b_0  b_1  .       b_s]
    # ---------------------------------------------------------------------------

    def __init__(self, domain, butcher_matrix, field_name=None,
                 fixed_subcycles=None, subcycle_by_courant=None,
                 increment_form=True, solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            butcher_matrix (numpy array): A matrix containing the coefficients of
                a butcher tableau defining a given Runge Kutta time discretisation.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            fixed_subcycles (int, optional): the fixed number of sub-steps to
                perform. This option cannot be specified with the
                `subcycle_by_courant` argument. Defaults to None.
            subcycle_by_courant (float, optional): specifying this option will
                make the scheme perform adaptive sub-cycling based on the
                Courant number. The specified argument is the maximum Courant
                for one sub-cycle. Defaults to None, in which case adaptive
                sub-cycling is not used. This option cannot be specified with the
                `fixed_subcycles` argument.
            increment_form (bool, optional): whether to write the RK scheme in
                "increment form", solving for increments rather than updated
                fields. Defaults to True.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        super().__init__(domain, field_name=field_name,
                         fixed_subcycles=fixed_subcycles,
                         subcycle_by_courant=subcycle_by_courant,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)
        self.butcher_matrix = butcher_matrix
        self.nStages = int(np.shape(self.butcher_matrix)[0])
        self.increment_form = increment_form

    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        super().setup(equation, apply_bcs, *active_labels)

        if not self.increment_form:
            self.field_i = [Function(self.fs) for i in range(self.nStages+1)]
        else:
            self.k = [Function(self.fs) for i in range(self.nStages)]

    @cached_property
    def solver(self):
        if self.increment_form:
            return super().solver
        else:
            # In this case, don't set snes_type to ksp only, as we do want the
            # outer Newton iteration
            solver_list = []

            for stage in range(self.nStages):
                # setup linear solver using lhs and rhs defined in derived class
                problem = NonlinearVariationalProblem(
                    self.lhs[stage].form - self.rhs[stage].form,
                    self.field_i[stage+1], bcs=self.bcs)
                solver_name = self.field_name+self.__class__.__name__+str(stage)
                solver = NonlinearVariationalSolver(
                    problem, solver_parameters=self.solver_parameters,
                    options_prefix=solver_name)
                solver_list.append(solver)
            return solver_list

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""

        if self.increment_form:
            l = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x_out, self.idx),
                map_if_false=drop)

            return l.form

        else:
            lhs_list = []
            for stage in range(self.nStages):
                l = self.residual.label_map(
                    lambda t: t.has_label(time_derivative),
                    map_if_true=replace_subject(self.field_i[stage+1], self.idx),
                    map_if_false=drop)
                lhs_list.append(l)

            return lhs_list

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side."""

        if self.increment_form:
            r = self.residual.label_map(
                all_terms,
                map_if_true=replace_subject(self.x1, old_idx=self.idx))

            r = r.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=drop,
                map_if_false=lambda t: -1*t)

            # If there are no active labels, we may have no terms at this point
            # So that we can still do xnp1 = xn, put in a zero term here
            if self.increment_form and len(r.terms) == 0:
                logger.warning('No terms detected for RHS of explicit problem. '
                               + 'Adding a zero term to avoid failure.')
                null_term = Constant(0.0)*self.residual.label_map(
                    lambda t: t.has_label(time_derivative),
                    # Drop label from this
                    map_if_true=lambda t: time_derivative.remove(t),
                    map_if_false=drop)
                r += null_term

            return r.form

        else:
            rhs_list = []

            for stage in range(self.nStages):
                r = self.residual.label_map(
                    all_terms,
                    map_if_true=replace_subject(self.field_i[0], old_idx=self.idx))

                r = r.label_map(
                    lambda t: t.has_label(time_derivative),
                    map_if_true=keep,
                    map_if_false=lambda t: -self.butcher_matrix[stage, 0]*self.dt*t)

                for i in range(1, stage+1):
                    r_i = self.residual.label_map(
                        lambda t: t.has_label(time_derivative),
                        map_if_true=drop,
                        map_if_false=replace_subject(self.field_i[i], old_idx=self.idx)
                    )

                    r -= self.butcher_matrix[stage, i]*self.dt*r_i

                rhs_list.append(r)

            return rhs_list

    def solve_stage(self, x0, stage):

        if self.increment_form:
            self.x1.assign(x0)

            for i in range(stage):
                self.x1.assign(self.x1 + self.dt*self.butcher_matrix[stage-1, i]*self.k[i])
            for evaluate in self.evaluate_source:
                evaluate(self.x1, self.dt)
            if self.limiter is not None:
                self.limiter.apply(self.x1)
            self.solver.solve()

            self.k[stage].assign(self.x_out)

            if (stage == self.nStages - 1):
                self.x1.assign(x0)
                for i in range(self.nStages):
                    self.x1.assign(self.x1 + self.dt*self.butcher_matrix[stage, i]*self.k[i])
                self.x1.assign(self.x1)

                if self.limiter is not None:
                    self.limiter.apply(self.x1)

        else:
            # Set initial field
            if stage == 0:
                self.field_i[0].assign(x0)

            # Use x0 as a first guess (otherwise may not converge)
            self.field_i[stage+1].assign(x0)

            # Update field_i for physics / limiters
            for evaluate in self.evaluate_source:
                # TODO: not implemented! Here we need to evaluate the m-th term
                # in the i-th RHS with field_m
                raise NotImplementedError(
                    'Physics not implemented with RK schemes that do not use '
                    + 'the increment form')
            if self.limiter is not None:
                self.limiter.apply(self.field_i[stage])

            # Obtain field_ip1 = field_n - dt* sum_m{a_im*F[field_m]}
            self.solver[stage].solve()

            if (stage == self.nStages - 1):
                self.x1.assign(self.field_i[stage+1])
                if self.limiter is not None:
                    self.limiter.apply(self.x1)

    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """
        if self.limiter is not None:
            self.limiter.apply(x_in)

        self.x1.assign(x_in)

        for i in range(self.nStages):
            self.solve_stage(x_in, i)
        x_out.assign(self.x1)


class ForwardEuler(ExplicitRungeKutta):
    """
    Implements the forward Euler timestepping scheme.

    The forward Euler method for operator F is the most simple explicit
    scheme:                                                                   \n
    k0 = F[y^n]                                                               \n
    y^(n+1) = y^n + dt*k0                                                     \n
    """
    def __init__(self, domain, field_name=None, fixed_subcycles=None,
                 subcycle_by_courant=None, increment_form=True,
                 solver_parameters=None, limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            fixed_subcycles (int, optional): the fixed number of sub-steps to
                perform. This option cannot be specified with the
                `subcycle_by_courant` argument. Defaults to None.
            subcycle_by_courant (float, optional): specifying this option will
                make the scheme perform adaptive sub-cycling based on the
                Courant number. The specified argument is the maximum Courant
                for one sub-cycle. Defaults to None, in which case adaptive
                sub-cycling is not used. This option cannot be specified with the
                `fixed_subcycles` argument.
            increment_form (bool, optional): whether to write the RK scheme in
                "increment form", solving for increments rather than updated
                fields. Defaults to True.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        butcher_matrix = np.array([1.]).reshape(1, 1)
        super().__init__(domain, butcher_matrix, field_name=field_name,
                         fixed_subcycles=fixed_subcycles,
                         subcycle_by_courant=subcycle_by_courant,
                         increment_form=increment_form,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class SSPRK3(ExplicitRungeKutta):
    u"""
    Implements the 3-stage Strong-Stability-Preserving Runge-Kutta method
    for solving ∂y/∂t = F(y). It can be written as:                           \n

    k0 = F[y^n]                                                               \n
    k1 = F[y^n + dt*k1]                                                       \n
    k2 = F[y^n + (1/4)*dt*(k0+k1)]                                            \n
    y^(n+1) = y^n + (1/6)*dt*(k0 + k1 + 4*k2)                                 \n
    """
    def __init__(self, domain, field_name=None, fixed_subcycles=None,
                 subcycle_by_courant=None, increment_form=True,
                 solver_parameters=None, limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            fixed_subcycles (int, optional): the fixed number of sub-steps to
                perform. This option cannot be specified with the
                `subcycle_by_courant` argument. Defaults to None.
            subcycle_by_courant (float, optional): specifying this option will
                make the scheme perform adaptive sub-cycling based on the
                Courant number. The specified argument is the maximum Courant
                for one sub-cycle. Defaults to None, in which case adaptive
                sub-cycling is not used. This option cannot be specified with the
                `fixed_subcycles` argument.
            increment_form (bool, optional): whether to write the RK scheme in
                "increment form", solving for increments rather than updated
                fields. Defaults to True.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        butcher_matrix = np.array([[1., 0., 0.], [1./4., 1./4., 0.], [1./6., 1./6., 2./3.]])

        super().__init__(domain, butcher_matrix, field_name=field_name,
                         fixed_subcycles=fixed_subcycles,
                         subcycle_by_courant=subcycle_by_courant,
                         increment_form=increment_form,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class RK4(ExplicitRungeKutta):
    u"""
    Implements the classic 4-stage Runge-Kutta method.

    The classic 4-stage Runge-Kutta method for solving ∂y/∂t = F(y). It can be
    written as:                                                               \n

    k0 = F[y^n]                                                               \n
    k1 = F[y^n + 1/2*dt*k1]                                                   \n
    k2 = F[y^n + 1/2*dt*k2]                                                   \n
    k3 = F[y^n + dt*k3]                                                       \n
    y^(n+1) = y^n + (1/6) * dt * (k0 + 2*k1 + 2*k2 + k3)                      \n

    where superscripts indicate the time-level.                               \n
    """
    def __init__(self, domain, field_name=None, fixed_subcycles=None,
                 subcycle_by_courant=None, increment_form=True,
                 solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            fixed_subcycles (int, optional): the fixed number of sub-steps to
                perform. This option cannot be specified with the
                `subcycle_by_courant` argument. Defaults to None.
            subcycle_by_courant (float, optional): specifying this option will
                make the scheme perform adaptive sub-cycling based on the
                Courant number. The specified argument is the maximum Courant
                for one sub-cycle. Defaults to None, in which case adaptive
                sub-cycling is not used. This option cannot be specified with the
                `fixed_subcycles` argument.
            increment_form (bool, optional): whether to write the RK scheme in
                "increment form", solving for increments rather than updated
                fields. Defaults to True.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        butcher_matrix = np.array([[0.5, 0., 0., 0.], [0., 0.5, 0., 0.], [0., 0., 1., 0.], [1./6., 1./3., 1./3., 1./6.]])
        super().__init__(domain, butcher_matrix, field_name=field_name,
                         fixed_subcycles=fixed_subcycles,
                         subcycle_by_courant=subcycle_by_courant,
                         increment_form=increment_form,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class Heun(ExplicitRungeKutta):
    u"""
    Implements Heun's method.

    The 2-stage Runge-Kutta scheme known as Heun's method,for solving
    ∂y/∂t = F(y). It can be written as:                                       \n

    y_1 = F[y^n]                                                              \n
    y^(n+1) = (1/2)y^n + (1/2)F[y_1]                                          \n

    where superscripts indicate the time-level and subscripts indicate the stage
    number.
    """
    def __init__(self, domain, field_name=None, fixed_subcycles=None,
                 subcycle_by_courant=None, increment_form=True,
                 solver_parameters=None, limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            fixed_subcycles (int, optional): the fixed number of sub-steps to
                perform. This option cannot be specified with the
                `subcycle_by_courant` argument. Defaults to None.
            subcycle_by_courant (float, optional): specifying this option will
                make the scheme perform adaptive sub-cycling based on the
                Courant number. The specified argument is the maximum Courant
                for one sub-cycle. Defaults to None, in which case adaptive
                sub-cycling is not used. This option cannot be specified with the
                `fixed_subcycles` argument.
            increment_form (bool, optional): whether to write the RK scheme in
                "increment form", solving for increments rather than updated
                fields. Defaults to True.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        butcher_matrix = np.array([[1., 0.], [0.5, 0.5]])
        super().__init__(domain, butcher_matrix, field_name=field_name,
                         fixed_subcycles=fixed_subcycles,
                         subcycle_by_courant=subcycle_by_courant,
                         increment_form=increment_form,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)
