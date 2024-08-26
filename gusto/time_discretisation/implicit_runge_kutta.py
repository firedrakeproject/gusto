"""Objects to describe implicit multi-stage (Runge-Kutta) discretisations."""

import numpy as np

from firedrake import (Function, split, NonlinearVariationalProblem,
                       NonlinearVariationalSolver)
from firedrake.fml import replace_subject, all_terms, drop
from firedrake.utils import cached_property

from gusto.core.labels import time_derivative
from gusto.time_discretisation.time_discretisation import TimeDiscretisation


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
    We compute the gradient at the intermediate location, k_i = F(y_i)        \n

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
                 solver_parameters=None, limiter=None, options=None,):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            butcher_matrix (numpy array): A matrix containing the coefficients
                of a butcher tableau defining a given Runge Kutta time
                discretisation.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
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
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)
        self.butcher_matrix = butcher_matrix
        self.nStages = int(np.shape(self.butcher_matrix)[1])

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

    def lhs(self):
        return super().lhs

    def rhs(self):
        return super().rhs

    def solver(self, stage):
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

        solver_name = self.field_name+self.__class__.__name__ + "%s" % (stage)
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    @cached_property
    def solvers(self):
        solvers = []
        for stage in range(self.nStages):
            solvers.append(self.solver(stage))
        return solvers

    def solve_stage(self, x0, stage):
        self.x1.assign(x0)
        for i in range(stage):
            self.x1.assign(self.x1 + self.butcher_matrix[stage, i]*self.dt*self.k[i])

        if self.limiter is not None:
            self.limiter.apply(self.x1)

        if self.idx is None and len(self.fs) > 1:
            self.xnph = tuple([self.dt*self.butcher_matrix[stage, stage]*a + b
                               for a, b in zip(split(self.x_out), split(self.x1))])
        else:
            self.xnph = self.x1 + self.butcher_matrix[stage, stage]*self.dt*self.x_out
        solver = self.solvers[stage]
        solver.solve()

        self.k[stage].assign(self.x_out)

    @wrapper_apply
    def apply(self, x_out, x_in):

        for i in range(self.nStages):
            self.solve_stage(x_in, i)

        x_out.assign(x_in)
        for i in range(self.nStages):
            x_out.assign(x_out + self.butcher_matrix[self.nStages, i]*self.dt*self.k[i])

        if self.limiter is not None:
            self.limiter.apply(x_out)


class ImplicitMidpoint(ImplicitRungeKutta):
    u"""
    Implements the Implicit Midpoint method as a 1-stage Runge Kutta method.

    The method, for solving
    ∂y/∂t = F(y), can be written as:                                          \n

    k0 = F[y^n + 0.5*dt*k0]                                                   \n
    y^(n+1) = y^n + dt*k0                                                     \n
    """
    def __init__(self, domain, field_name=None, solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        butcher_matrix = np.array([[0.5], [1.]])
        super().__init__(domain, butcher_matrix, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class QinZhang(ImplicitRungeKutta):
    u"""
    Implements Qin and Zhang's two-stage, 2nd order, implicit Runge–Kutta method.

    The method, for solving
    ∂y/∂t = F(y), can be written as:                                          \n

    k0 = F[y^n + 0.25*dt*k0]                                                  \n
    k1 = F[y^n + 0.5*dt*k0 + 0.25*dt*k1]                                      \n
    y^(n+1) = y^n + 0.5*dt*(k0 + k1)                                          \n
    """
    def __init__(self, domain, field_name=None, solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        butcher_matrix = np.array([[0.25, 0], [0.5, 0.25], [0.5, 0.5]])
        super().__init__(domain, butcher_matrix, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)
