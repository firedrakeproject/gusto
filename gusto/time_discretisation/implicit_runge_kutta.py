"""Objects to describe implicit multi-stage (Runge-Kutta) discretisations."""

import numpy as np

from firedrake import (Function, split, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, Constant)
from firedrake.fml import replace_subject, all_terms, drop
from firedrake.utils import cached_property

from gusto.core.labels import time_derivative, source_label
from gusto.time_discretisation.time_discretisation import (
    TimeDiscretisation, wrapper_apply
)
from gusto.time_discretisation.explicit_runge_kutta import RungeKuttaFormulation


__all__ = ["ImplicitRungeKutta", "ImplicitMidpoint", "QinZhang"]


class ImplicitRungeKutta(TimeDiscretisation):
    """
    A class for implementing general diagonally implicit multistage (Runge-Kutta)
    methods based on its Butcher tableau.

    Unlike the explicit method, all upper diagonal a_ij elements are non-zero
    for implicit methods.

    There are three steps to move from the current solution, y^n, to the new
    one, y^{n+1}

    For each i = 1, s  in an s stage method
    we have the intermediate solutions:                                       \n
    y_i = y^n +  dt*(a_i1*k_1 + a_i2*k_2 + ... + a_ii*k_i)                    \n
    For the increment form we compute the gradient at the                     \n
    intermediate location, k_i = F(y_i), whilst for the                       \n
    predictor form we solve for each intermediate solution y_i.               \n

    At the last stage, compute the new solution by:                           \n
    y^{n+1} = y^n + dt*(b_1*k_1 + b_2*k_2 + .... + b_s*k_s)
    """
    # ---------------------------------------------------------------------------
    # Butcher tableau for a s-th order
    # diagonally implicit scheme:
    #  c_0 | a_00  0    .     0
    #  c_1 | a_10 a_11  .     0
    #   .  |   .   .    .     .
    #   .  |   .   .    .     .
    #  c_s | a_s0 a_s1  .    a_ss
    #   -------------------------
    #      |  b_1  b_2  ...  b_s
    #
    #
    # The corresponding square 'butcher_matrix' is:
    #
    #  [a_00   0   .       0  ]
    #  [a_10 a_11  .       0  ]
    #  [  .    .   .       .  ]
    #  [ b_0  b_1  .       b_s]
    # ---------------------------------------------------------------------------

    def __init__(self, domain, butcher_matrix, field_name=None,
                 rk_formulation=RungeKuttaFormulation.increment,
                 solver_parameters=None, options=None, augmentation=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            butcher_matrix (numpy array): A matrix containing the coefficients
                of a butcher tableau defining a given Runge Kutta time
                discretisation.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            rk_formulation (:class:`RungeKuttaFormulation`, optional):
                an enumerator object, describing the formulation of the Runge-
                Kutta scheme. Defaults to the increment form.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        super().__init__(domain, field_name=field_name,
                         solver_parameters=solver_parameters,
                         options=options, augmentation=augmentation)
        self.butcher_matrix = butcher_matrix
        self.nStages = int(np.shape(self.butcher_matrix)[1])
        self.rk_formulation = rk_formulation

    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """

        super().setup(equation, apply_bcs, *active_labels)

        self.k = [Function(self.fs) for i in range(self.nStages)]

        # Check that we do not have source terms
        for t in self.residual:
            if (t.has_label(source_label)):
                raise NotImplementedError("Source terms have not been implemented with implicit RK schemes")

        if self.rk_formulation == RungeKuttaFormulation.predictor:
            self.xs = [Function(self.fs) for _ in range(self.nStages)]
        elif self.rk_formulation == RungeKuttaFormulation.increment:
            self.k = [Function(self.fs) for _ in range(self.nStages)]
        elif self.rk_formulation == RungeKuttaFormulation.linear:
            raise NotImplementedError(
                'Linear Implicit Runge-Kutta formulation is not implemented'
            )
        else:
            raise NotImplementedError(
                'Runge-Kutta formulation is not implemented'
            )

    def res(self, stage):
        """Set up the residual for the predictor formulation for a given stage."""
        # Add time derivative terms  y_s - y^n for stage s
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)
        residual = mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        residual -= mass_form.label_map(all_terms,
                                        map_if_true=replace_subject(self.x1, old_idx=self.idx))
        # Loop through stages up to s-1 and calculate sum
        # dt*(a_s1*F(y_1) + a_s2*F(y_2)+ ... + a_{s,s-1}*F(y_{s-1}))
        for i in range(stage):
            r_imp = self.residual.label_map(
                lambda t: not t.has_label(time_derivative),
                map_if_true=replace_subject(self.xs[i], old_idx=self.idx),
                map_if_false=drop)
            r_imp = r_imp.label_map(
                all_terms,
                map_if_true=lambda t: Constant(self.butcher_matrix[stage, i])*self.dt*t)
            residual += r_imp
        # Calculate and add on dt*a_ss*F(y_s)
        r_imp = self.residual.label_map(
            lambda t: not t.has_label(time_derivative),
            map_if_true=replace_subject(self.x_out, old_idx=self.idx),
            map_if_false=drop)
        r_imp = r_imp.label_map(
            all_terms,
            map_if_true=lambda t: Constant(self.butcher_matrix[stage, stage])*self.dt*t)
        residual += r_imp
        return residual.form

    @property
    def final_res(self):
        """Set up the final residual for the predictor formulation."""
        # Add time derivative terms  y^{n+1} - y^n
        mass_form = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                            map_if_false=drop)
        residual = mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        residual -= mass_form.label_map(all_terms,
                                        map_if_true=replace_subject(self.x1, old_idx=self.idx))
        # Loop through stages up to s-1 and calculate/sum
        # dt*(b_1*F(y_1) + b_2*F(y_2) + .... + b_s*F(y_s))
        for i in range(self.nStages):
            r_imp = self.residual.label_map(
                lambda t: not t.has_label(time_derivative),
                map_if_true=replace_subject(self.xs[i], old_idx=self.idx),
                map_if_false=drop)
            r_imp = r_imp.label_map(
                all_terms,
                map_if_true=lambda t: Constant(self.butcher_matrix[self.nStages, i])*self.dt*t)
            residual += r_imp
        return residual.form

    def solver(self, stage):
        if self.rk_formulation == RungeKuttaFormulation.increment:
            residual = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=drop,
                map_if_false=replace_subject(self.xnph, self.idx),
            )
            mass_form = self.residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_false=drop)
            residual += mass_form.label_map(all_terms,
                                            replace_subject(self.x_out, self.idx))

            problem = NonlinearVariationalProblem(residual.form, self.x_out, bcs=self.bcs)

        elif self.rk_formulation == RungeKuttaFormulation.predictor:
            problem = NonlinearVariationalProblem(self.res(stage), self.x_out, bcs=self.bcs)

        solver_name = self.field_name+self.__class__.__name__ + "%s" % (stage)
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @cached_property
    def final_solver(self):
        """
        Set up a solver for the final solve for the predictor
        formulation to evaluate time level n+1.
        """
        # setup solver using residual (res) defined in derived class
        problem = NonlinearVariationalProblem(self.final_res, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @cached_property
    def solvers(self):
        solvers = []
        for stage in range(self.nStages):
            solvers.append(self.solver(stage))
        return solvers

    def solve_stage(self, x0, stage):
        self.x1.assign(x0)
        if self.rk_formulation == RungeKuttaFormulation.increment:
            for i in range(stage):
                self.x1.assign(self.x1 + self.butcher_matrix[stage, i]*self.dt*self.k[i])

            if self.idx is None and len(self.fs) > 1:
                self.xnph = tuple(
                    self.dt * self.butcher_matrix[stage, stage] * a + b
                    for a, b in zip(split(self.x_out), split(self.x1))
                )
            else:
                self.xnph = self.x1 + self.butcher_matrix[stage, stage]*self.dt*self.x_out

            solver = self.solvers[stage]

            # Set initial guess for solver
            if (stage > 0):
                self.x_out.assign(self.k[stage-1])

            solver.solve()
            self.k[stage].assign(self.x_out)

        elif self.rk_formulation == RungeKuttaFormulation.predictor:
            if (stage > 0):
                self.x_out.assign(self.xs[stage-1])
            solver = self.solvers[stage]
            solver.solve()

            self.xs[stage].assign(self.x_out)

    @wrapper_apply
    def apply(self, x_out, x_in):
        self.x_out.assign(x_in)
        for i in range(self.nStages):
            self.solve_stage(x_in, i)

        if self.rk_formulation == RungeKuttaFormulation.increment:
            x_out.assign(x_in)
            for i in range(self.nStages):
                x_out.assign(x_out + self.butcher_matrix[self.nStages, i]*self.dt*self.k[i])
        elif self.rk_formulation == RungeKuttaFormulation.predictor:
            self.final_solver.solve()
            x_out.assign(self.x_out)


class ImplicitMidpoint(ImplicitRungeKutta):
    u"""
    Implements the Implicit Midpoint method as a 1-stage Runge Kutta method.

    The method, for solving
    ∂y/∂t = F(y), can be written as:                                          \n

    k0 = F[y^n + 0.5*dt*k0]                                                   \n
    y^(n+1) = y^n + dt*k0                                                     \n
    """
    def __init__(self, domain, field_name=None,
                 rk_formulation=RungeKuttaFormulation.increment,
                 solver_parameters=None, options=None, augmentation=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            rk_formulation (:class:`RungeKuttaFormulation`, optional):
                an enumerator object, describing the formulation of the Runge-
                Kutta scheme. Defaults to the increment form.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        butcher_matrix = np.array([[0.5], [1.]])
        super().__init__(domain, butcher_matrix, field_name,
                         rk_formulation=rk_formulation,
                         solver_parameters=solver_parameters,
                         options=options, augmentation=augmentation)


class QinZhang(ImplicitRungeKutta):
    u"""
    Implements Qin and Zhang's two-stage, 2nd order, implicit Runge–Kutta method.

    The method, for solving
    ∂y/∂t = F(y), can be written as:                                          \n

    k0 = F[y^n + 0.25*dt*k0]                                                  \n
    k1 = F[y^n + 0.5*dt*k0 + 0.25*dt*k1]                                      \n
    y^(n+1) = y^n + 0.5*dt*(k0 + k1)                                          \n
    """
    def __init__(self, domain, field_name=None,
                 rk_formulation=RungeKuttaFormulation.increment,
                 solver_parameters=None, options=None, augmentation=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            rk_formulation (:class:`RungeKuttaFormulation`, optional):
                an enumerator object, describing the formulation of the Runge-
                Kutta scheme. Defaults to the increment form.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        butcher_matrix = np.array([[0.25, 0], [0.5, 0.25], [0.5, 0.5]])
        super().__init__(domain, butcher_matrix, field_name,
                         rk_formulation=rk_formulation,
                         solver_parameters=solver_parameters,
                         options=options, augmentation=augmentation)
