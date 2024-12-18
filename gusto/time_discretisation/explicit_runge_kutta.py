"""Objects to describe explicit multi-stage (Runge-Kutta) discretisations."""

import numpy as np

from enum import Enum
from firedrake import (Function, Constant, NonlinearVariationalProblem,
                       NonlinearVariationalSolver)
from firedrake.fml import replace_subject, all_terms, drop, keep, Term
from firedrake.utils import cached_property
from firedrake.formmanipulation import split_form

from gusto.core.labels import time_derivative, all_but_last
from gusto.core.logging import logger
from gusto.time_discretisation.time_discretisation import ExplicitTimeDiscretisation


__all__ = [
    "ForwardEuler", "ExplicitRungeKutta", "SSPRK3", "RK4", "Heun",
    "RungeKuttaFormulation"
]


class RungeKuttaFormulation(Enum):
    """
    Enumerator to describe the formulation of a Runge-Kutta scheme.

    The following Runge-Kutta methods for solving dy/dt = F(y) are encoded here:
    - `increment`:                                                            \n
        k_0 = F[y^n]                                                          \n
        k_m = F[y^n - dt*Sum_{i=0}^{m-1} a_{m,i} * k_i], for m = 1 to M - 1   \n
        y^{n+1} = y^n - dt*Sum_{i=0}^{M-1} b_i*k_i                            \n
    - `predictor`:
        y^0 = y^n                                                             \n
        y^m = y^0 - dt*Sum_{i=0}^{m-1} a_{m,i} * F[y^i], for m = 1 to M - 1   \n
        y^{n+1} = y^0 - dt*Sum_{i=0}^{m-1} b_i * F[y^i]                       \n
    - `linear`:
        y^0 = y^n                                                             \n
        y^m = y^0 - dt*F[Sum_{i=0}^{m-1} a_{m,i} * y^i], for m = 1 to M - 1   \n
        y^{n+1} = y^0 - dt*F[Sum_{i=0}^{m-1} b_i * y^i]                       \n
    """

    increment = 1595712
    predictor = 8234900
    linear = 269207


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
                 subcycling_options=None,
                 rk_formulation=RungeKuttaFormulation.increment,
                 solver_parameters=None, limiter=None, options=None,
                 augmentation=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            butcher_matrix (numpy array): A matrix containing the coefficients
                of a butcher tableau defining a given Runge Kutta scheme.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
            rk_formulation (:class:`RungeKuttaFormulation`, optional):
                an enumerator object, describing the formulation of the Runge-
                Kutta scheme. Defaults to the increment form.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        super().__init__(domain, field_name=field_name,
                         subcycling_options=subcycling_options,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options,
                         augmentation=augmentation)
        self.butcher_matrix = butcher_matrix
        self.nbutcher = int(np.shape(self.butcher_matrix)[0])
        self.rk_formulation = rk_formulation

    @property
    def nStages(self):
        return self.nbutcher

    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        super().setup(equation, apply_bcs, *active_labels)

        if self.rk_formulation == RungeKuttaFormulation.predictor:
            self.field_i = [Function(self.fs) for _ in range(self.nStages+1)]
        elif self.rk_formulation == RungeKuttaFormulation.increment:
            self.k = [Function(self.fs) for _ in range(self.nStages)]
        elif self.rk_formulation == RungeKuttaFormulation.linear:
            self.field_rhs = Function(self.fs)
        else:
            raise NotImplementedError(
                'Runge-Kutta formulation is not implemented'
            )

    @cached_property
    def solver(self):
        if self.rk_formulation == RungeKuttaFormulation.increment:
            return super().solver

        elif self.rk_formulation == RungeKuttaFormulation.predictor:
            solver_list = []

            for stage in range(self.nStages):
                # setup linear solver using lhs and rhs defined in derived class
                problem = NonlinearVariationalProblem(
                    self.lhs[stage].form - self.rhs[stage].form,
                    self.field_i[stage+1], bcs=self.bcs
                )
                solver_name = self.field_name+self.__class__.__name__+str(stage)
                solver = NonlinearVariationalSolver(
                    problem, solver_parameters=self.solver_parameters,
                    options_prefix=solver_name
                )
                solver_list.append(solver)
            return solver_list

        elif self.rk_formulation == RungeKuttaFormulation.linear:
            problem = NonlinearVariationalProblem(
                self.lhs - self.rhs[0], self.x1, bcs=self.bcs
            )
            solver_name = self.field_name+self.__class__.__name__
            solver = NonlinearVariationalSolver(
                problem, solver_parameters=self.solver_parameters,
                options_prefix=solver_name
            )

            # Set up problem for final step
            problem_last = NonlinearVariationalProblem(
                self.lhs - self.rhs[1], self.x1, bcs=self.bcs
            )
            solver_name = self.field_name+self.__class__.__name__+'_last'
            solver_last = NonlinearVariationalSolver(
                problem_last, solver_parameters=self.solver_parameters,
                options_prefix=solver_name
            )

            return solver, solver_last

        else:
            raise NotImplementedError(
                'Runge-Kutta formulation is not implemented'
            )

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""

        if self.rk_formulation == RungeKuttaFormulation.increment:
            l = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x_out, old_idx=self.idx),
                map_if_false=drop)

            return l.form

        elif self.rk_formulation == RungeKuttaFormulation.predictor:
            lhs_list = []
            for stage in range(self.nStages):
                l = self.residual.label_map(
                    lambda t: t.has_label(time_derivative),
                    map_if_true=replace_subject(self.field_i[stage+1], old_idx=self.idx),
                    map_if_false=drop)
                lhs_list.append(l)

            return lhs_list

        if self.rk_formulation == RungeKuttaFormulation.linear:
            l = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x1, old_idx=self.idx),
                map_if_false=drop)

            return l.form

        else:
            raise NotImplementedError(
                'Runge-Kutta formulation is not implemented'
            )

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side."""

        if self.rk_formulation == RungeKuttaFormulation.increment:
            r = self.residual.label_map(
                all_terms,
                map_if_true=replace_subject(self.x1, old_idx=self.idx))

            r = r.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=drop,
                map_if_false=lambda t: -1*t)

            # If there are no active labels, we may have no terms at this point
            # So that we can still do xnp1 = xn, put in a zero term here
            if len(r.terms) == 0:
                logger.warning('No terms detected for RHS of explicit problem. '
                               + 'Adding a zero term to avoid failure.')
                null_term = Constant(0.0)*self.residual.label_map(
                    lambda t: t.has_label(time_derivative),
                    # Drop label from this
                    map_if_true=lambda t: time_derivative.remove(t),
                    map_if_false=drop)
                r += null_term

            return r.form

        elif self.rk_formulation == RungeKuttaFormulation.predictor:
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

        elif self.rk_formulation == RungeKuttaFormulation.linear:

            r = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x0, old_idx=self.idx),
                map_if_false=replace_subject(self.field_rhs, old_idx=self.idx)
            )
            r = r.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=keep,
                map_if_false=lambda t: -self.dt*t
            )

            # Set up all-but-last RHS
            if self.idx is not None:
                # If original function is in mixed function space, then ensure
                # correct test function in the all-but-last form
                r_all_but_last = self.residual.label_map(
                    lambda t: t.has_label(all_but_last),
                    map_if_true=lambda t:
                        Term(split_form(t.get(all_but_last).form)[self.idx].form,
                             t.labels),
                    map_if_false=keep
                )
            else:
                r_all_but_last = self.residual.label_map(
                    lambda t: t.has_label(all_but_last),
                    map_if_true=lambda t: Term(t.get(all_but_last).form, t.labels),
                    map_if_false=keep
                )
            r_all_but_last = r_all_but_last.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(self.x0, old_idx=self.idx),
                map_if_false=replace_subject(self.field_rhs, old_idx=self.idx)
            )
            r_all_but_last = r_all_but_last.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=keep,
                map_if_false=lambda t: -self.dt*t
            )

            return r_all_but_last.form, r.form

        else:
            raise NotImplementedError(
                'Runge-Kutta formulation is not implemented'
            )

    def solve_stage(self, x0, stage):

        if self.rk_formulation == RungeKuttaFormulation.increment:
            self.x1.assign(x0)

            for i in range(stage):
                self.x1.assign(self.x1 + self.dt*self.butcher_matrix[stage-1, i]*self.k[i])
            for evaluate in self.evaluate_source:
                evaluate(self.x1, self.dt)
            if self.limiter is not None:
                self.limiter.apply(self.x1)

            # Set initial guess for solver
            if stage > 0:
                self.x_out.assign(self.k[stage-1])
            self.solver.solve()

            self.k[stage].assign(self.x_out)

            if (stage == self.nStages - 1):
                self.x1.assign(x0)
                for i in range(self.nStages):
                    self.x1.assign(self.x1 + self.dt*self.butcher_matrix[stage, i]*self.k[i])
                self.x1.assign(self.x1)

                if self.limiter is not None:
                    self.limiter.apply(self.x1)

        elif self.rk_formulation == RungeKuttaFormulation.predictor:
            # Set initial field
            if stage == 0:
                self.field_i[0].assign(x0)

            # Use previous stage value as a first guess (otherwise may not converge)
            self.field_i[stage+1].assign(self.field_i[stage])

            # Update field_i for physics / limiters
            for evaluate in self.evaluate_source:
                # TODO: not implemented! Here we need to evaluate the m-th term
                # in the i-th RHS with field_m
                raise NotImplementedError(
                    'Physics not implemented with RK schemes that use the '
                    + 'predictor form')
            if self.limiter is not None:
                self.limiter.apply(self.field_i[stage])

            # Obtain field_ip1 = field_n - dt* sum_m{a_im*F[field_m]}
            self.solver[stage].solve()

            if (stage == self.nStages - 1):
                self.x1.assign(self.field_i[stage+1])
                if self.limiter is not None:
                    self.limiter.apply(self.x1)

        elif self.rk_formulation == RungeKuttaFormulation.linear:

            # Set combined index of stage and subcycle
            cycle_stage = self.nStages*self.subcycle_idx + stage

            if stage == 0 and self.subcycle_idx == 0:
                self.field_lhs = [Function(self.fs) for _ in range(self.nStages*self.ncycles)]
                self.field_lhs[0].assign(self.x0)

            # All-but-last form ------------------------------------------------
            if (cycle_stage + 1 < self.ncycles*self.nStages):
                # Build up RHS field to be evaluated
                self.field_rhs.assign(0.0)
                for i in range(stage+1):
                    i_cycle_stage = self.nStages*self.subcycle_idx + i
                    self.field_rhs.assign(
                        self.field_rhs
                        + self.butcher_matrix[stage, i]*self.field_lhs[i_cycle_stage]
                    )

                # Evaluate physics and apply limiter, if necessary
                for evaluate in self.evaluate_source:
                    evaluate(self.field_rhs, self.dt)
                if self.limiter is not None:
                    self.limiter.apply(self.field_rhs)

                # Use previous stage value as a first guess (otherwise may not converge)
                self.x1.assign(self.field_lhs[cycle_stage])
                # Solve problem, placing solution in self.x1
                self.solver[0].solve()

                # Store LHS
                self.field_lhs[cycle_stage+1].assign(self.x1)

            # Last stage and last subcycle -------------------------------------
            else:
                # Build up RHS field to be evaluated
                self.field_rhs.assign(0.0)
                for i in range(self.ncycles*self.nStages):
                    j = i % self.nStages
                    self.field_rhs.assign(
                        self.field_rhs
                        + self.butcher_matrix[self.nStages-1, j]*self.field_lhs[i]
                    )

                # Evaluate physics and apply limiter, if necessary
                for evaluate in self.evaluate_source:
                    evaluate(self.field_rhs, self.original_dt)
                if self.limiter is not None:
                    self.limiter.apply(self.field_rhs)
                # Use x0 as a first guess (otherwise may not converge)
                self.x1.assign(x0)
                # Solve problem, placing solution in self.x1
                self.solver[1].solve()

                # Final application of limiter
                if self.limiter is not None:
                    self.limiter.apply(self.x1)

        else:
            raise NotImplementedError(
                'Runge-Kutta formulation is not implemented'
            )

    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_in (:class:`Function`): the input field.
            x_out (:class:`Function`): the output field to be computed.
        """

        if self.augmentation is not None:
            self.augmentation.update(x_in)

        # TODO: is this limiter application necessary?
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
    def __init__(
            self, domain, field_name=None, subcycling_options=None,
            rk_formulation=RungeKuttaFormulation.increment,
            solver_parameters=None, limiter=None, options=None,
            augmentation=None
    ):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
            rk_formulation (:class:`RungeKuttaFormulation`, optional):
                an enumerator object, describing the formulation of the Runge-
                Kutta scheme. Defaults to the increment form.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """

        butcher_matrix = np.array([1.]).reshape(1, 1)

        super().__init__(domain, butcher_matrix, field_name=field_name,
                         subcycling_options=subcycling_options,
                         rk_formulation=rk_formulation,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options,
                         augmentation=augmentation)


class SSPRK3(ExplicitRungeKutta):
    u"""
    Implements 3rd order Strong-Stability-Preserving Runge-Kutta methods
    for solving ∂y/∂t = F(y).                                                 \n

    The 3-stage method can be written as:                                     \n

    k0 = F[y^n]                                                               \n
    k1 = F[y^n + dt*k1]                                                       \n
    k2 = F[y^n + (1/4)*dt*(k0+k1)]                                            \n
    y^(n+1) = y^n + (1/6)*dt*(k0 + k1 + 4*k2)                                 \n

    The 4-stage method can be written as:                                     \n

    k0 = F[y^n]                                                               \n
    k1 = F[y^n + (1/2)*dt*k1]                                                 \n
    k2 = F[y^n + (1/2)*dt*(k0+k1)]                                            \n
    k3 = F[y^n + (1/6)*dt*(k0+k1+k2)]                                         \n
    y^(n+1) = y^n + (1/6)*dt*(k0 + k1 + k2 + 3*k3)                            \n

    The 5-stage method can be written as:                                     \n

    k0 = F[y^n]                                                               \n
    k1 = F[y^n + (1/2)*dt*k1]                                                 \n
    k2 = F[y^n + (1/2)*dt*(k0+k1)]                                            \n
    k3 = F[y^n + (1/6)*dt*(k0+k1+k2)]                                         \n
    y^(n+1) = y^n + (1/6)*dt*(k0 + k1 + k2 + 3*k3)                            \n
    """
    def __init__(
            self, domain, field_name=None, subcycling_options=None,
            rk_formulation=RungeKuttaFormulation.increment,
            solver_parameters=None, limiter=None, options=None,
            augmentation=None, stages=3
    ):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
            rk_formulation (:class:`RungeKuttaFormulation`, optional):
                an enumerator object, describing the formulation of the Runge-
                Kutta scheme. Defaults to the increment form.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
            stages (int, optional): number of stages: (3, 4, 5). Defaults to 3.
        """

        if stages == 3:
            butcher_matrix = np.array([
                [1., 0., 0.],
                [1./4., 1./4., 0.],
                [1./6., 1./6., 2./3.]
            ])
            self.cfl_limit = 1
        elif stages == 4:
            butcher_matrix = np.array([
                [1./2., 0., 0., 0.],
                [1./2., 1./2., 0., 0.],
                [1./6., 1./6., 1./6., 0.],
                [1./6., 1./6., 1./6., 1./2.]
            ])
            self.cfl_limit = 2
        elif stages == 5:
            self.cfl_limit = 2.65062919294483
            butcher_matrix = np.array([
                [0.37726891511710, 0., 0., 0., 0.],
                [0.37726891511710, 0.37726891511710, 0., 0., 0.],
                [0.16352294089771, 0.16352294089771, 0.16352294089771, 0., 0.],
                [0.14904059394856, 0.14831273384724, 0.14831273384724, 0.34217696850008, 0.],
                [0.19707596384481, 0.11780316509765, 0.11709725193772, 0.27015874934251, 0.29786487010104]
            ])
        else:
            raise ValueError(f"{stages} stage 3rd order SSPRK not implemented")

        super().__init__(domain, butcher_matrix, field_name=field_name,
                         subcycling_options=subcycling_options,
                         rk_formulation=rk_formulation,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options,
                         augmentation=augmentation)


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
    def __init__(
            self, domain, field_name=None, subcycling_options=None,
            rk_formulation=RungeKuttaFormulation.increment,
            solver_parameters=None, limiter=None, options=None,
            augmentation=None
    ):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
            rk_formulation (:class:`RungeKuttaFormulation`, optional):
                an enumerator object, describing the formulation of the Runge-
                Kutta scheme. Defaults to the increment form.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        butcher_matrix = np.array([
            [0.5, 0., 0., 0.],
            [0., 0.5, 0., 0.],
            [0., 0., 1., 0.],
            [1./6., 1./3., 1./3., 1./6.]
        ])
        super().__init__(domain, butcher_matrix, field_name=field_name,
                         subcycling_options=subcycling_options,
                         rk_formulation=rk_formulation,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options,
                         augmentation=augmentation)


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
    def __init__(
            self, domain, field_name=None, subcycling_options=None,
            rk_formulation=RungeKuttaFormulation.increment,
            solver_parameters=None, limiter=None, options=None,
            augmentation=None
    ):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
            rk_formulation (:class:`RungeKuttaFormulation`, optional):
                an enumerator object, describing the formulation of the Runge-
                Kutta scheme. Defaults to the increment form.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """

        butcher_matrix = np.array([
            [1., 0.],
            [0.5, 0.5]
        ])
        super().__init__(domain, butcher_matrix, field_name=field_name,
                         subcycling_options=subcycling_options,
                         rk_formulation=rk_formulation,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options,
                         augmentation=augmentation)
