"""Implementations of IMEX Runge-Kutta time discretisations."""

from firedrake import (Function, Constant, NonlinearVariationalProblem,
                       NonlinearVariationalSolver)
from firedrake.fml import replace_subject, all_terms, drop
from firedrake.utils import cached_property
from gusto.core.labels import time_derivative, implicit, explicit, physics_label
from gusto.time_discretisation.time_discretisation import (
    TimeDiscretisation, wrapper_apply
)
import numpy as np


__all__ = ["IMEXRungeKutta", "IMEX_Euler", "IMEX_ARS3", "IMEX_ARK2",
           "IMEX_Trap2", "IMEX_SSP3"]


class IMEXRungeKutta(TimeDiscretisation):
    """
    A class for implementing general IMEX multistage (Runge-Kutta)
    methods based on two Butcher tableaus, to solve                           \n

    ∂y/∂t = F(y) + S(y)                                                       \n

    Where F are implicit fast terms, and S are explicit slow terms.           \n

    There are three steps to move from the current solution, y^n, to the new
    one, y^{n+1}                                                              \n

    For each i = 1, s  in an s stage method
    we compute the intermediate solutions:                                    \n
    y_i = y^n + dt*(a_i1*F(y_1) + a_i2*F(y_2)+ ... + a_ii*F(y_i))             \n
              + dt*(d_i1*S(y_1) + d_i2*S(y_2)+ ... + d_{i,i-1}*S(y_{i-1}))

    At the last stage, compute the new solution by:                           \n
    y^{n+1} = y^n + dt*(b_1*F(y_1) + b_2*F(y_2) + .... + b_s*F(y_s))          \n
                  + dt*(e_1*S(y_1) + e_2*S(y_2) + .... + e_s*S(y_s))          \n

    """
    # --------------------------------------------------------------------------
    # Butcher tableaus for a s-th order
    # diagonally implicit scheme (left) and explicit scheme (right):
    #  c_0 | a_00  0    .     0        f_0 |   0   0    .     0
    #  c_1 | a_10 a_11  .     0        f_1 | d_10  0    .     0
    #   .  |   .   .    .     .         .  |   .   .    .     .
    #   .  |   .   .    .     .         .  |   .   .    .     .
    #  c_s | a_s0 a_s1  .    a_ss      f_s | d_s0 d_s1  .     0
    #   -------------------------       -------------------------
    #      |  b_1  b_2  ...  b_s           |  b_1  b_2  ...  b_s
    #
    #
    # The corresponding square 'butcher_imp' and 'butcher_exp' matrices are:
    #
    #  [a_00   0   0   .   0  ]        [  0    0   0   .   0  ]
    #  [a_10 a_11  0   .   0  ]        [d_10   0   0   .   0  ]
    #  [a_20 a_21 a_22 .   0  ]        [d_20  d_21 0   .   0  ]
    #  [  .    .   .   .   .  ]        [  .    .   .   .   .  ]
    #  [ b_0  b_1  .       b_s]        [ e_0  e_1  .   .   e_s]
    #
    # --------------------------------------------------------------------------

    def __init__(self, domain, butcher_imp, butcher_exp, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            butcher_imp (:class:`numpy.ndarray`): A matrix containing the
                coefficients of a butcher tableau defining a given implicit
                Runge Kutta time discretisation.
            butcher_exp (:class:`numpy.ndarray`): A matrix containing the
                coefficients of a butcher tableau defining a given explicit
                Runge Kutta time discretisation.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            linear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying linear solver. Defaults to None.
            nonlinear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying nonlinear solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        super().__init__(domain, field_name=field_name,
                         solver_parameters=nonlinear_solver_parameters,
                         options=options)
        self.butcher_imp = butcher_imp
        self.butcher_exp = butcher_exp
        self.nStages = int(np.shape(self.butcher_imp)[1])

        # Set default linear and nonlinear solver options if none passed in
        if linear_solver_parameters is None:
            self.linear_solver_parameters = {'snes_type': 'ksponly',
                                             'ksp_type': 'cg',
                                             'pc_type': 'bjacobi',
                                             'sub_pc_type': 'ilu'}
        else:
            self.linear_solver_parameters = linear_solver_parameters

        if nonlinear_solver_parameters is None:
            self.nonlinear_solver_parameters = {'snes_type': 'newtonls',
                                                'ksp_type': 'gmres',
                                                'pc_type': 'bjacobi',
                                                'sub_pc_type': 'ilu'}
        else:
            self.nonlinear_solver_parameters = nonlinear_solver_parameters

    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """

        super().setup(equation, apply_bcs, *active_labels)

        # Check all terms are labeled implicit, exlicit
        for t in self.residual:
            if ((not t.has_label(implicit)) and (not t.has_label(explicit))
               and (not t.has_label(time_derivative)) and (not t.has_label(physics_label))):
                raise NotImplementedError("Non time-derivative terms must be labeled as implicit or explicit")

        self.xs = [Function(self.fs) for i in range(self.nStages)]

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(IMEXRungeKutta, self).lhs

    @cached_property
    def rhs(self):
        """Set up the discretisation's right hand side (the time derivative)."""
        return super(IMEXRungeKutta, self).rhs

    def res(self, stage):
        """Set up the discretisation's residual for a given stage."""
        # Add time derivative terms  y_s - y^n for stage s
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)
        residual = mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        residual -= mass_form.label_map(all_terms,
                                        map_if_true=replace_subject(self.x1, old_idx=self.idx))
        # Loop through stages up to s-1 and calcualte/sum
        # dt*(a_s1*F(y_1) + a_s2*F(y_2)+ ... + a_{s,s-1}*F(y_{s-1}))
        # and
        # dt*(d_s1*S(y_1) + d_s2*S(y_2)+ ... + d_{s,s-1}*S(y_{s-1}))
        for i in range(stage):
            r_exp = self.residual.label_map(
                lambda t: t.has_label(explicit),
                map_if_true=replace_subject(self.xs[i], old_idx=self.idx),
                map_if_false=drop)
            r_exp = r_exp.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_false=lambda t: Constant(self.butcher_exp[stage, i])*self.dt*t)
            r_imp = self.residual.label_map(
                lambda t: t.has_label(implicit),
                map_if_true=replace_subject(self.xs[i], old_idx=self.idx),
                map_if_false=drop)
            r_imp = r_imp.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_false=lambda t: Constant(self.butcher_imp[stage, i])*self.dt*t)
            residual += r_imp
            residual += r_exp
        # Calculate and add on dt*a_ss*F(y_s)
        r_imp = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.x_out, old_idx=self.idx),
            map_if_false=drop)
        r_imp = r_imp.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: Constant(self.butcher_imp[stage, stage])*self.dt*t)
        residual += r_imp
        return residual.form

    @property
    def final_res(self):
        """Set up the discretisation's final residual."""
        # Add time derivative terms  y^{n+1} - y^n
        mass_form = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                            map_if_false=drop)
        residual = mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        residual -= mass_form.label_map(all_terms,
                                        map_if_true=replace_subject(self.x1, old_idx=self.idx))
        # Loop through stages up to s-1 and calcualte/sum
        # dt*(b_1*F(y_1) + b_2*F(y_2) + .... + b_s*F(y_s))
        # and
        # dt*(e_1*S(y_1) + e_2*S(y_2) + .... + e_s*S(y_s))
        for i in range(self.nStages):
            r_exp = self.residual.label_map(
                lambda t: t.has_label(explicit),
                map_if_true=replace_subject(self.xs[i], old_idx=self.idx),
                map_if_false=drop)
            r_exp = r_exp.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_false=lambda t: Constant(self.butcher_exp[self.nStages, i])*self.dt*t)
            r_imp = self.residual.label_map(
                lambda t: t.has_label(implicit),
                map_if_true=replace_subject(self.xs[i], old_idx=self.idx),
                map_if_false=drop)
            r_imp = r_imp.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_false=lambda t: Constant(self.butcher_imp[self.nStages, i])*self.dt*t)
            residual += r_imp
            residual += r_exp
        return residual.form

    @cached_property
    def solvers(self):
        """Set up a list of solvers for each problem at a stage."""
        solvers = []
        for stage in range(self.nStages):
            # setup solver using residual defined in derived class
            problem = NonlinearVariationalProblem(self.res(stage), self.x_out, bcs=self.bcs)
            solver_name = self.field_name+self.__class__.__name__ + "%s" % (stage)
            solvers.append(NonlinearVariationalSolver(problem, solver_parameters=self.nonlinear_solver_parameters, options_prefix=solver_name))
        return solvers

    @cached_property
    def final_solver(self):
        """Set up a solver for the final solve to evaluate time level n+1."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.final_res, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.linear_solver_parameters, options_prefix=solver_name)

    @wrapper_apply
    def apply(self, x_out, x_in):
        self.x1.assign(x_in)
        self.x_out.assign(x_in)
        solver_list = self.solvers

        for stage in range(self.nStages):
            self.solver = solver_list[stage]
            # Set initial solver guess
            if (stage > 0):
                self.x_out.assign(self.xs[stage-1])
            self.solver.solve()

            # Apply limiter
            if self.limiter is not None:
                self.limiter.apply(self.x_out)
            self.xs[stage].assign(self.x_out)

        self.final_solver.solve()

        # Apply limiter
        if self.limiter is not None:
            self.limiter.apply(self.x_out)
        x_out.assign(self.x_out)


class IMEX_Euler(IMEXRungeKutta):
    u"""
    Implements IMEX Euler one-stage method.

    The method, for solving                                                   \n
    ∂y/∂t = F(y) + S(y), can be written as:                                   \n

    y_0 = y^n                                                                 \n
    y_1 = y^n + dt*F[y_1] + dt*S[y_0]                                         \n
    y^(n+1) = y^n + dt*F[y_1] + dt*S[y_0]
    """
    def __init__(self, domain, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            linear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying linear solver. Defaults to None.
            nonlinear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying nonlinear solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        butcher_imp = np.array([[0., 0.], [0., 1.], [0., 1.]])
        butcher_exp = np.array([[0., 0.], [1., 0.], [1., 0.]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options)


class IMEX_ARS3(IMEXRungeKutta):
    u"""
    Implements ARS3(2,3,3) two-stage IMEX Runge–Kutta method
    from RK IMEX for HEVI (Weller et al 2013).
    Where g = (3 + sqrt(3))/6.

    The method, for solving                                                   \n
    ∂y/∂t = F(y) + S(y), can be written as:                                   \n

    y_0 = y^n                                                                 \n
    y_1 = y^n + dt*g*F[y_1] + dt*g*S[y_0]                                     \n
    y_2 = y^n + dt*((1-2g)*F[y_1]+g*F[y_2])                                   \n
              + dt*((g-1)*S[y_0]+2(g-1)*S[y_1])                               \n
    y^(n+1) = y^n + dt*(g*F[y_1]+(1-g)*F[y_2])                                \n
                  + dt*(0.5*S[y_1]+0.5*S[y_2])
    """
    def __init__(self, domain, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            linear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying linear solver. Defaults to None.
            nonlinear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying nonlinear solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        g = (3. + np.sqrt(3.))/6.
        butcher_imp = np.array([[0., 0., 0.], [0., g, 0.], [0., 1-2.*g, g], [0., 0.5, 0.5]])
        butcher_exp = np.array([[0., 0., 0.], [g, 0., 0.], [g-1., 2.*(1.-g), 0.], [0., 0.5, 0.5]])

        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options)


class IMEX_ARK2(IMEXRungeKutta):
    u"""
    Implements ARK2(2,3,2) two-stage IMEX Runge–Kutta method from
    RK IMEX for HEVI (Weller et al 2013).
    Where g = 1 - 1/sqrt(2), a = 1/6(3 + 2sqrt(2)), d = 1/2sqrt(2).

    The method, for solving                                                   \n
    ∂y/∂t = F(y) + S(y), can be written as:                                   \n

    y_0 = y^n                                                                 \n
    y_1 = y^n + dt*(g*F[y_0]+g*F[y_1]) + 2*dt*g*S[y_0]                        \n
    y_2 = y^n + dt*(d*F[y_0]+d*F[y_1]+g*F[y_2])                               \n
              + dt*((1-a)*S[y_0]+a*S[y_1])                                    \n
    y^(n+1) = y^n + dt*(d*F[y_0]+d*F[y_1]+g*F[y_2])                           \n
                  + dt*(d*S[y_0]+d*S[y_1]+g*S[y_2])
    """
    def __init__(self, domain, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            linear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying linear solver. Defaults to None.
            nonlinear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying nonlinear solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        g = 1. - 1./np.sqrt(2.)
        d = 1./(2.*np.sqrt(2.))
        a = 1./6.*(3. + 2.*np.sqrt(2.))
        butcher_imp = np.array([[0., 0., 0.], [g, g, 0.], [d, d, g], [d, d, g]])
        butcher_exp = np.array([[0., 0., 0.], [2.*g, 0., 0.], [1.-a, a, 0.], [d, d, g]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options)


class IMEX_SSP3(IMEXRungeKutta):
    u"""
    Implements SSP3(3,3,2) three-stage IMEX Runge–Kutta method from RK IMEX for
    HEVI (Weller et al 2013).

    Let g = 1 - 1/sqrt(2). The method, for solving                            \n
    ∂y/∂t = F(y) + S(y), can be written as:                                   \n

    y_1 = y^n + dt*g*F[y_1]                                                   \n
    y_2 = y^n + dt*((1-2g)*F[y_1]+g*F[y_2]) + dt*S[y_1]                       \n
    y_3 = y^n + dt*((0.5-g)*F[y_1]+g*F[y_3]) + dt*(0.25*S[y_1]+0.25*S[y_2])   \n
    y^(n+1) = y^n + dt*(1/6*F[y_1]+1/6*F[y_2]+2/3*F[y_3])                     \n
                  + dt*(1/6*S[y_1]+1/6*S[y_2]+2/3*S[y_3])
    """
    def __init__(self, domain, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            linear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying linear solver. Defaults to None.
            nonlinear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying nonlinear solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        g = 1. - (1./np.sqrt(2.))
        butcher_imp = np.array([[g, 0., 0.], [1-2.*g, g, 0.], [0.5-g, 0., g], [(1./6.), (1./6.), (2./3.)]])
        butcher_exp = np.array([[0., 0., 0.], [1., 0., 0.], [0.25, 0.25, 0.], [(1./6.), (1./6.), (2./3.)]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options)


class IMEX_Trap2(IMEXRungeKutta):
    u"""
    Implements Trap2(2+e,3,2) three-stage IMEX Runge–Kutta method from RK IMEX for HEVI (Weller et al 2013).
    For e = 1 or 0.

    The method, for solving                                                    \n
    ∂y/∂t = F(y) + S(y), can be written as:                                    \n

    y_0 = y^n                                                                  \n
    y_1 = y^n + dt*e*F[y_0] + dt*S[y_0]                                        \n
    y_2 = y^n + dt*(0.5*F[y_0]+0.5*F[y_2]) + dt*(0.5*S[y_0]+0.5*S[y_1])        \n
    y_3 = y^n + dt*(0.5*F[y_0]+0.5*F[y_3]) + dt*(0.5*S[y_0]+0.5*S[y_2])        \n
    y^(n+1) = y^n + dt*(0.5*F[y_0]+0.5*F[y_3]) + dt*(0.5*S[y_0] + 0.5*S[y_2])  \n
    """
    def __init__(self, domain, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            linear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying linear solver. Defaults to None.
            nonlinear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying nonlinear solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        e = 0.
        butcher_imp = np.array([[0., 0., 0., 0.], [e, 0., 0., 0.], [0.5, 0., 0.5, 0.], [0.5, 0., 0., 0.5], [0.5, 0., 0., 0.5]])
        butcher_exp = np.array([[0., 0., 0., 0.], [1., 0., 0., 0.], [0.5, 0.5, 0., 0.], [0.5, 0., 0.5, 0.], [0.5, 0., 0.5, 0.]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options)
