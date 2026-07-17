"""Implementations of IMEX Runge-Kutta time discretisations."""

from functools import cached_property
from firedrake import (Cofunction, Function, Constant, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, assemble, derivative)
from firedrake.fml import replace_subject, all_terms, drop
from gusto.core.labels import time_derivative, implicit, explicit, source_label
from gusto.time_discretisation.time_discretisation import (
    TimeDiscretisation, wrapper_apply
)
import numpy as np
from qmat.qcoeff.butcher import ARK548L2SAESDIRK2, ARK548L2SAERK2
from gusto.solvers.solver_presets import hybridised_solver_parameters


__all__ = ["IMEXRungeKutta", "IMEX_Euler", "IMEX_ARS3", "IMEX_ARK2",
           "IMEX_Trap2", "IMEX_SSP3", "IMEX_ARS443", "IMEX_ARK4", "IMEX_ARK5"]


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
             limiter=None, options=None, augmentation=None, multiple_solvers=False):
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
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
            multiple_solvers (bool, optional): If True, use a separate solver for
                each stage. If False, use a single solver for all stages.
        """
        super().__init__(domain, field_name=field_name,
                         solver_parameters=nonlinear_solver_parameters,
                         options=options, augmentation=augmentation)
        self.butcher_imp = butcher_imp
        self.butcher_exp = butcher_exp
        self.nStages = int(np.shape(self.butcher_imp)[1])

        # Some butcher tableaus have zero first stage, if so, we don't need to do an
        # initial solve and can copy across x_in to x_s[0]
        self.zero_first_stage = True
        self.solver_start_stage = 1
        for value in self.butcher_imp[0]:
            if value != 0.0:
                self.zero_first_stage = False
                self.solver_start_stage = 0

        # Set default linear and nonlinear solver options if none passed in
        if linear_solver_parameters is None:
            self.linear_solver_parameters = {'snes_type': 'ksponly',
                                             'ksp_type': 'cg',
                                             'pc_type': 'bjacobi',
                                             'sub_pc_type': 'ilu'}
        else:
            self.linear_solver_parameters = linear_solver_parameters

        # Set default linear and nonlinear solver options if none passed in
        if linear_solver_parameters is None:
            self.linear_solver_parameters = {'snes_type': 'ksponly',
                                             'ksp_type': 'cg',
                                             'pc_type': 'bjacobi',
                                             'sub_pc_type': 'ilu'}
        else:
            self.linear_solver_parameters = linear_solver_parameters

        self.nonlinear_solver_parameters = nonlinear_solver_parameters

        self.multiple_solvers = multiple_solvers
        self.total_ksp_its = 0
        self._step_count = 1
        self.lag_rebuild_freq = self.nonlinear_solver_parameters.get("td_lag_rebuild", None)
        if self.lag_rebuild_freq is not None:
            if self.lag_rebuild_freq < 1:
                raise ValueError("IMEXRungeKutta: lag_rebuild_freq must be >= 1")
            elif not isinstance(self.lag_rebuild_freq, int):
                raise ValueError("IMEXRungeKutta: lag_rebuild_freq must be an integer")
            else:
                from gusto import logger
                logger.info(f"IMEXRungeKutta: lag_rebuild_freq set to {self.lag_rebuild_freq}. "
                            "Jacobian will be rebuilt every lag_rebuild_freq timesteps.")

    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """

        super().setup(equation, apply_bcs, *active_labels)

        self.equation = equation

        # Check all terms are labeled implicit, exlicit
        for t in self.residual:
            if ((not t.has_label(implicit)) and (not t.has_label(explicit))
               and (not t.has_label(time_derivative)) and (not t.has_label(source_label))):
                raise NotImplementedError("Non time-derivative or source terms must be labeled as implicit or explicit")

        self.xs = [Function(self.fs) for i in range(self.nStages)]
        self.source = [Function(self.fs) for i in range(self.nStages)]
        self.b = Cofunction(self.fs.dual())
        # Check whether the implicit diagonal is constant. The single shared
        # solver reuses one operator (I - alpha*dt*F) across all stages, which is
        # only valid if every implicit stage has the same diagonal coefficient.
        diag = np.diag(self.butcher_imp[:self.nStages, :self.nStages])
        nz = diag[diag != 0.0]
        self._constant_diag = (nz.size == 0) or bool(np.allclose(nz, nz[0]))

        self.appctx=None

        if not self._constant_diag and not self.multiple_solvers:
            from gusto import logger
            logger.warning(
                "IMEXRungeKutta: implicit diagonal is not constant "
                f"(diag={diag}); the shared single-solver path is invalid. "
                "Falling back to multiple per-stage solvers.")
            self.multiple_solvers = True

        self.alpha = Constant(self.butcher_imp[self.nStages-1, self.nStages-1])

        if not self.multiple_solvers and self.nonlinear_solver_parameters is None:
            # Use hybridised solver as default
            self.nonlinear_solver_parameters, self.appctx = hybridised_solver_parameters(self.equation, self.equation.field_names, alpha=alpha, tau_values=None, nonlinear=True, imex=True)


    def _lag_reset(self, solvers):
        if self.lag_rebuild_freq is None:
            return
        rebuild = ((self._step_count - 1) % self.lag_rebuild_freq == 0)
        for s in solvers:
            s._ctx._jacobian_assembled = not rebuild
    
    def res_mult(self, stage):
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

            # Calculate source terms
            r_source = self.residual.label_map(
                lambda t: t.has_label(source_label),
                map_if_true=replace_subject(self.source[i], old_idx=self.idx),
                map_if_false=drop)
            r_source = r_source.label_map(
                all_terms,
                map_if_true=lambda t: Constant(self.butcher_exp[stage, i]) * self.dt * t
            )
            residual += r_source

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

    def res(self, stage):
        """Set up the discretisation's residual if the diagonal coefficients are the same."""
        # Add time derivative terms  y_s - y^n for stage s
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)
        residual = -mass_form.label_map(all_terms,
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

            # Calculate source terms
            r_source = self.residual.label_map(
                lambda t: t.has_label(source_label),
                map_if_true=replace_subject(self.source[i], old_idx=self.idx),
                map_if_false=drop)
            r_source = r_source.label_map(
                all_terms,
                map_if_true=lambda t: Constant(self.butcher_exp[stage, i]) * self.dt * t
            )
            residual += r_source

        return residual.form

    @cached_property
    def stage_rhs(self):
        """Cached stage RHS forms.

        The form *structure* is fixed once; the coefficients it references
        (self.x1, self.xs[i], self.source[i]) are updated in place by .assign
        each step, so the cached forms stay valid across timesteps. Removes the
        per-step label_map/replace_subject rebuild that res(stage) otherwise
        repeats every apply().
        """
        return [self.res(stage)
                for stage in range(self.solver_start_stage, self.nStages)]
    
    def resval(self):
        """Set up the discretisation's residual for a given stage."""
        # Add time derivative terms  y_s - y^n for stage s
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)

        residual = mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        # Calculate and add on dt*a_ss*F(y_s)
        r_imp = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.x_out, old_idx=self.idx),
            map_if_false=drop)
        r_imp = r_imp.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: self.alpha*self.dt*t)
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
            # Calculate source terms
            r_source = self.residual.label_map(
                lambda t: t.has_label(source_label),
                map_if_true=replace_subject(self.source[i], old_idx=self.idx),
                map_if_false=drop)
            r_source = r_source.label_map(
                all_terms,
                map_if_true=lambda t: Constant(self.butcher_exp[self.nStages, i])*self.dt*t)
            residual += r_source
        return residual.form

    @cached_property
    def solvers(self):
        """Set up a list of solvers for each problem at a stage."""
        solvers = []
        for stage in range(self.solver_start_stage, self.nStages):
            # setup solver using residual defined in derived class
            if self.nonlinear_solver_parameters is None:
                alpha = self.butcher_imp[stage, stage]
                self.nonlinear_solver_parameters, self.appctx = hybridised_solver_parameters(self.equation, self.equation.field_names, alpha=alpha, tau_values=None, nonlinear=True, imex=True)
            problem = NonlinearVariationalProblem(self.res_mult(stage), self.x_out, bcs=self.bcs)
            problem._constant_jacobian = True
            solver_name = self.field_name+self.__class__.__name__ + "%s" % (stage)
            solvers.append(NonlinearVariationalSolver(problem, solver_parameters=self.nonlinear_solver_parameters, appctx=self.appctx, options_prefix=solver_name))
        return solvers
    
    @cached_property
    def solver(self):   
        """Set up a solver for the shared problem at a stage."""
        F = self.resval() + self.b
        J = derivative(self.resval(), self.x_out)
        problem = NonlinearVariationalProblem(F, self.x_out, bcs=self.bcs, J=J)
        if self.lag_rebuild_freq is not None:
            problem._constant_jacobian = True
        name = self.field_name + self.__class__.__name__ + "shared"
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=self.nonlinear_solver_parameters,
            options_prefix=name)

        return solver
    @cached_property
    def final_solver(self):
        """Set up a solver for the final solve to evaluate time level n+1."""
        # setup solver using residual (res) defined in derived class
        problem = NonlinearVariationalProblem(self.final_res, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.linear_solver_parameters, options_prefix=solver_name)

    @wrapper_apply
    def apply(self, x_out, x_in):
        from firedrake import PETSc
        self.x1.assign(x_in)
        self.x_out.assign(x_in)
        self.xs[0].assign(x_in)

        if self.multiple_solvers:
            solvers_list = self.solvers
        else:
            solvers_list = [self.solver]

        self._lag_reset(solvers_list)

        for stage in range(self.solver_start_stage, self.nStages):
            if stage != self.solver_start_stage:
                self.x_out.assign(self.xs[stage-1])
            for evaluate in self.evaluate_source:
                evaluate(self.xs[stage-1], self.dt, x_out=self.source[stage-1])

            if self.multiple_solvers:
                solver = solvers_list[stage-self.solver_start_stage]
                solver.solve()
                self.total_ksp_its += solver.snes.getLinearSolveIterations()
            else:
                assemble(self.stage_rhs[stage-self.solver_start_stage], tensor=self.b)
                self.solver.solve()
                self.total_ksp_its += self.solver.snes.getLinearSolveIterations()

            if self.limiter is not None:
                self.limiter.apply(self.x_out)
            self.xs[stage].assign(self.x_out)

        for evaluate in self.evaluate_source:
            evaluate(self.xs[-1], self.dt, x_out=self.source[-1])
        self.final_solver.solve()

        if self.limiter is not None:
            self.limiter.apply(self.x_out)
        x_out.assign(self.x_out)
        self._step_count = self._step_count + 1


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
                 limiter=None, options=None, augmentation=None):
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
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        butcher_imp = np.array([[0., 0.], [0., 1.], [0., 1.]])
        butcher_exp = np.array([[0., 0.], [1., 0.], [1., 0.]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options, augmentation=augmentation)


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
                 limiter=None, options=None, augmentation=None):
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
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        g = (3. + np.sqrt(3.))/6.
        butcher_imp = np.array([[0., 0., 0.], [0., g, 0.], [0., 1-2.*g, g], [0., 0.5, 0.5]])
        butcher_exp = np.array([[0., 0., 0.], [g, 0., 0.], [g-1., 2.*(1.-g), 0.], [0., 0.5, 0.5]])

        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options, augmentation=augmentation)


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
                 limiter=None, options=None, augmentation=None):
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
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        g = 1. - 1./np.sqrt(2.)
        d = 1./(2.*np.sqrt(2.))
        a = 1./6.*(3. + 2.*np.sqrt(2.))
        butcher_imp = np.array([[0., 0., 0.], [g, g, 0.], [d, d, g], [d, d, g]])
        butcher_exp = np.array([[0., 0., 0.], [2.*g, 0., 0.], [1.-a, a, 0.], [d, d, g]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options, augmentation=augmentation)


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
                 limiter=None, options=None, augmentation=None):
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
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        g = 1. - (1./np.sqrt(2.))
        butcher_imp = np.array([[g, 0., 0.], [1-2.*g, g, 0.], [0.5-g, 0., g], [(1./6.), (1./6.), (2./3.)]])
        butcher_exp = np.array([[0., 0., 0.], [1., 0., 0.], [0.25, 0.25, 0.], [(1./6.), (1./6.), (2./3.)]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options, augmentation=augmentation)


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
                 limiter=None, options=None, augmentation=None):
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
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        e = 0.
        butcher_imp = np.array([[0., 0., 0., 0.], [e, 0., 0., 0.], [0.5, 0., 0.5, 0.], [0.5, 0., 0., 0.5], [0.5, 0., 0., 0.5]])
        butcher_exp = np.array([[0., 0., 0., 0.], [1., 0., 0., 0.], [0.5, 0.5, 0., 0.], [0.5, 0., 0.5, 0.], [0.5, 0., 0.5, 0.]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options, augmentation=augmentation)

class IMEX_ARS443(IMEXRungeKutta):
    r"""
    Implements ARS(4,4,3) IMEX Runge–Kutta method (Ascher–Ruuth–Spiteri 1997).
    """

    def __init__(self, domain, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None, augmentation=None):

        # ---------- Explicit tableau (ERK) ----------
        butcher_exp = np.array([
            # a_ij for stages 1..4
            [0., 0., 0., 0.],
            [1767732205903/2027836641118, 0., 0., 0.],
            [5535828885825/10492691773637, 788022342437/10882634858940, 0., 0.],
            [6485989280629/16251701735622,
             -4246266847089/9704473918619,
              10755448449292/10357097424841, 0.],
            # ---- b (weights)
            [1471266399579/7840856788654,
             -4482444167858/7529755066697,
              11266239266428/11593286722821,
              1767732205903/4055673282236]
        ], dtype=float)

        # ---------- Implicit tableau (DIRK) ----------
        g = 1767732205903/4055673282236

        butcher_imp = np.array([
            [0., 0., 0., 0.],
            [g,   g,  0., 0.],
            [2746238789719/10658868560708,
            -640167445237/6845629431997,
            g, 0.],
            [1471266399579/7840856788654,
            -4482444167858/7529755066697,
            11266239266428/11593286722821,
            g],      # This is also bI
            [1471266399579/7840856788654,
            -4482444167858/7529755066697,
            11266239266428/11593286722821,
            g],  
        ], dtype=float)

        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options, augmentation=augmentation)
        
class IMEX_ARK4(IMEXRungeKutta):
    r"""
    Implements ARK4(3)6L[2]SA (Kennedy–Carpenter) 4th‑order IMEX Runge–Kutta.
    """

    def __init__(self, domain, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None, augmentation=None):

        # ---------- Explicit tableau (ERK) ----------
        Aexp = [
            [0, 0, 0, 0, 0, 0],
            [1/2, 0, 0, 0, 0, 0],
            [13861/62500, 6889/62500, 0, 0, 0, 0],
            [-116923316275/2393684061468,
             -2731218467317/15368042101831,
              9408046702089/11113171139209, 0, 0, 0],
            [-451086348788/2902428689909,
             -2682348792572/7519795681897,
              12662868775082/11960479115383,
              3355817975965/11060851509271, 0, 0],
            [647845179188/3216320057751,
             73281519250/8382639484533,
             552539513391/3454668386233,
             3354512671639/8306763924573,
             4040/17871, 0],
            # --- b row
            [82889/524892, 0,
             15625/83664,
             69875/102672,
             -2260/8211,
             1/4]
        ]
        butcher_exp = np.array(Aexp, dtype=float)

        # ---------- Implicit tableau (DIRK) ----------
        Adirk = [
            [0, 0, 0, 0, 0, 0],
            [1/4, 1/4, 0, 0, 0, 0],
            [8611/62500, -1743/31250, 1/4, 0, 0, 0],
            [5012029/34652500, -654441/2922500, 174375/388108, 1/4, 0, 0],
            [15267082809/155376265600,
             -71443401/120774400,
              730878875/902184768,
              2285395/8070912,
              1/4, 0],
            [82889/524892, 0,
             15625/83664,
             69875/102672,
             -2260/8211,
              1/4],
            # stiffly‑accurate → b = last row
            [82889/524892, 0,
             15625/83664,
             69875/102672,
             -2260/8211,
             1/4]
        ]
        butcher_imp = np.array(Adirk, dtype=float)

        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         linear_solver_parameters=linear_solver_parameters,
                         nonlinear_solver_parameters=nonlinear_solver_parameters,
                         limiter=limiter, options=options, augmentation=augmentation)
        
class IMEX_ARK5(IMEXRungeKutta):
    r"""
    Implements ARK5(4)8L[2]SA IMEX Runge–Kutta scheme
    (Kennedy & Carpenter). This is the 5th-order additive RK pair
    known in SUNDIALS/ARKODE as ARK548L2SA (explicit + implicit).

    Explicit tableau:  ARK548L2SA_ERK
    Implicit tableau:  ARK548L2SA_DIRK (ESDIRK, γ = 1/4)
    """

    def __init__(self, domain, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None, augmentation=None):

        
        dirk = ARK548L2SAESDIRK2()
        erk = ARK548L2SAERK2()
        A_imp  = dirk.A
        A_exp  = erk.A
        b_imp = dirk.b
        b_exp = erk.b
        b_imp_row = b_imp.reshape(1, -1)
        b_exp_row = b_exp.reshape(1, -1)
        self.butcher_imp = np.vstack([A_imp, b_imp_row])
        self.butcher_exp = np.vstack([A_exp, b_exp_row])
        super().__init__(
            domain,
            self.butcher_imp, self.butcher_exp,
            field_name,
            linear_solver_parameters=linear_solver_parameters,
            nonlinear_solver_parameters=nonlinear_solver_parameters,
            limiter=limiter,
            options=options,
            augmentation=augmentation
        )