u"""
Objects for discretising time derivatives.

Time discretisation objects discretise ∂y/∂t = F(y), for variable y, time t and
operator F.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
import math
import numpy as np
from firedrake import (
    Function, TestFunction, TestFunctions, NonlinearVariationalProblem,
    NonlinearVariationalSolver, DirichletBC, split, Constant
)
from firedrake.fml import (
    replace_subject, replace_test_function, Term, all_terms, drop
)
from firedrake.formmanipulation import split_form
from firedrake.utils import cached_property

from gusto.configuration import EmbeddedDGOptions, RecoveryOptions
from gusto.labels import (time_derivative, prognostic, physics_label,
                          implicit, explicit)
from gusto.logging import logger, DEBUG, logging_ksp_monitor_true_residual
from gusto.wrappers import *
import scipy
from scipy.special import legendre
from firedrake.petsc import PETSc


__all__ = ["ForwardEuler", "BackwardEuler", "ExplicitMultistage",
           "IMEXMultistage", "SSPRK3", "RK4", "Heun", "ThetaMethod",
           "TrapeziumRule", "BDF2", "TR_BDF2", "Leapfrog", "AdamsMoulton",
           "AdamsBashforth", "ImplicitMidpoint", "QinZhang",
           "IMEX_Euler", "ARS3", "ARK2", "Trap2", "SSP3", "BE_SDC", "FE_SDC",
           "IMEX_SDC"]


def wrapper_apply(original_apply):
    """Decorator to add steps for using a wrapper around the apply method."""
    def get_apply(self, x_out, x_in):

        if self.wrapper is not None:

            def new_apply(self, x_out, x_in):

                self.wrapper.pre_apply(x_in)
                original_apply(self, self.wrapper.x_out, self.wrapper.x_in)
                self.wrapper.post_apply(x_out)

            return new_apply(self, x_out, x_in)

        else:

            return original_apply(self, x_out, x_in)

    return get_apply


class TimeDiscretisation(object, metaclass=ABCMeta):
    """Base class for time discretisation schemes."""

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
        self.domain = domain
        self.field_name = field_name
        self.equation = None

        self.dt = Constant(0.0)
        self.dt.assign(domain.dt)
        self.original_dt = Constant(0.0)
        self.original_dt.assign(self.dt)
        self.options = options
        self.limiter = limiter
        self.courant_max = None

        if options is not None:
            self.wrapper_name = options.name
            if self.wrapper_name == "mixed_options":
                self.wrapper = MixedFSWrapper()

                for field, suboption in options.suboptions.items():
                    if suboption.name == 'embedded_dg':
                        self.wrapper.subwrappers.update({field: EmbeddedDGWrapper(self, suboption)})
                    elif suboption.name == "recovered":
                        self.wrapper.subwrappers.update({field: RecoveryWrapper(self, suboption)})
                    elif suboption.name == "supg":
                        raise RuntimeError(
                            'Time discretisation: suboption SUPG is currently not implemented within MixedOptions')
                    else:
                        raise RuntimeError(
                            f'Time discretisation: suboption wrapper {wrapper_name} not implemented')
            elif self.wrapper_name == "embedded_dg":
                self.wrapper = EmbeddedDGWrapper(self, options)
            elif self.wrapper_name == "recovered":
                self.wrapper = RecoveryWrapper(self, options)
            elif self.wrapper_name == "supg":
                self.wrapper = SUPGWrapper(self, options)
            else:
                raise RuntimeError(
                    f'Time discretisation: wrapper {self.wrapper_name} not implemented')
        else:
            self.wrapper = None
            self.wrapper_name = None

        # get default solver options if none passed in
        if solver_parameters is None:
            self.solver_parameters = {'ksp_type': 'cg',
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}
        else:
            self.solver_parameters = solver_parameters

    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            apply_bcs (bool, optional): whether to apply the equation's boundary
                conditions. Defaults to True.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        self.equation = equation
        self.residual = equation.residual

        if self.field_name is not None and hasattr(equation, "field_names"):
            self.idx = equation.field_names.index(self.field_name)
            self.fs = equation.spaces[self.idx]
            self.residual = self.residual.label_map(
                lambda t: t.get(prognostic) == self.field_name,
                lambda t: Term(
                    split_form(t.form)[self.idx].form,
                    t.labels),
                drop)

        else:
            self.field_name = equation.field_name
            self.fs = equation.function_space
            self.idx = None

        bcs = equation.bcs[self.field_name]

        if len(active_labels) > 0:
            self.residual = self.residual.label_map(
                lambda t: any(t.has_label(time_derivative, *active_labels)),
                map_if_false=drop)

        self.evaluate_source = []
        self.physics_names = []
        for t in self.residual:
            if t.has_label(physics_label):
                physics_name = t.get(physics_label)
                if t.labels[physics_name] not in self.physics_names:
                    self.evaluate_source.append(t.labels[physics_name])
                    self.physics_names.append(t.labels[physics_name])

        # -------------------------------------------------------------------- #
        # Set up Wrappers
        # -------------------------------------------------------------------- #

        if self.wrapper is not None:
            if self.wrapper_name == "mixed_options":

                self.wrapper.wrapper_spaces = equation.spaces
                self.wrapper.field_names = equation.field_names

                for field, subwrapper in self.wrapper.subwrappers.items():

                    if field not in equation.field_names:
                        raise ValueError(f"The option defined for {field} is for a field that does not exist in the equation set")

                    field_idx = equation.field_names.index(field)
                    subwrapper.setup(equation.spaces[field_idx])

                    # Update the function space to that needed by the wrapper
                    self.wrapper.wrapper_spaces[field_idx] = subwrapper.function_space

                self.wrapper.setup()
                self.fs = self.wrapper.function_space
                new_test_mixed = TestFunctions(self.fs)

                # Replace the original test function with one from the new
                # function space defined by the subwrappers
                self.residual = self.residual.label_map(
                    all_terms,
                    map_if_true=replace_test_function(new_test_mixed))

            else:
                if self.wrapper_name == "supg":
                    self.wrapper.setup()
                else:
                    self.wrapper.setup(self.fs)
                self.fs = self.wrapper.function_space
                if self.solver_parameters is None:
                    self.solver_parameters = self.wrapper.solver_parameters
                new_test = TestFunction(self.wrapper.test_space)
                # SUPG has a special wrapper
                if self.wrapper_name == "supg":
                    new_test = self.wrapper.test

                # Replace the original test function with the one from the wrapper
                self.residual = self.residual.label_map(
                    all_terms,
                    map_if_true=replace_test_function(new_test))

                self.residual = self.wrapper.label_terms(self.residual)

        # -------------------------------------------------------------------- #
        # Make boundary conditions
        # -------------------------------------------------------------------- #

        if not apply_bcs:
            self.bcs = None
        elif self.wrapper is not None:
            # Transfer boundary conditions onto test function space
            self.bcs = [DirichletBC(self.fs, bc.function_arg, bc.sub_domain)
                        for bc in bcs]
        else:
            self.bcs = bcs

        # -------------------------------------------------------------------- #
        # Make the required functions
        # -------------------------------------------------------------------- #

        self.x_out = Function(self.fs)
        self.x1 = Function(self.fs)

    @property
    def nlevels(self):
        return 1

    @abstractproperty
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x_out, old_idx=self.idx),
            map_if_false=drop)

        return l.form

    @abstractproperty
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, old_idx=self.idx))

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: -self.dt*t)

        return r.form

    @cached_property
    def solver(self):
        """Set up the problem and the solver."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        solver = NonlinearVariationalSolver(
            problem,
            solver_parameters=self.solver_parameters,
            options_prefix=solver_name
        )
        if logger.isEnabledFor(DEBUG):
            solver.snes.ksp.setMonitor(logging_ksp_monitor_true_residual)
        return solver

    @abstractmethod
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        pass


class IMEXMultistage(TimeDiscretisation):
    """
    A class for implementing general IMEX multistage (Runge-Kutta)
    methods based on two Butcher tableaus, to solve                           \n

    ∂y/∂t = F(y) + S(y)                                                       \n

    Where F are implicit fast terms, and S are explicit slow terms.           \n

    There are three steps to move from the current solution, y^n, to the new one, y^{n+1}

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
                 solver_parameters=None, limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            butcher_imp (:class:`numpy.ndarray`): A matrix containing the coefficients of
                a butcher tableau defining a given implicit Runge Kutta time discretisation.
            butcher_exp (:class:`numpy.ndarray`): A matrix containing the coefficients of
                a butcher tableau defining a given explicit Runge Kutta time discretisation.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        super().__init__(domain, field_name=field_name,
                         solver_parameters=solver_parameters,
                         options=options)
        self.butcher_imp = butcher_imp
        self.butcher_exp = butcher_exp
        self.nStages = int(np.shape(self.butcher_imp)[1])

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
               and (not t.has_label(time_derivative))):
                raise NotImplementedError("Non time-derivative terms must be labeled as implicit or explicit")

        self.xs = [Function(self.fs) for i in range(self.nStages)]

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(IMEXMultistage, self).lhs

    @cached_property
    def rhs(self):
        """Set up the discretisation's right hand side (the time derivative)."""
        return super(IMEXMultistage, self).rhs

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
            solver_parameters = {'ksp_type': 'gmres',
                                'pc_type': 'bjacobi',
                                'ksp_rtol': 1e-3,
                                'ksp_atol': 1e-4,
                                'sub_pc_type': 'ilu'}
            solvers.append(NonlinearVariationalSolver(problem,  solver_parameters=solver_parameters, options_prefix=solver_name))
        return solvers

    @cached_property
    def final_solver(self):
        """Set up a solver for the final solve to evaluate time level n+1."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.final_res, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    def apply(self, x_out, x_in):
        self.x1.assign(x_in)
        self.x_out.assign(x_in)
        solver_list = self.solvers

        for stage in range(self.nStages):
            self.solver = solver_list[stage]
            self.solver.solve()
            self.xs[stage].assign(self.x_out)

        self.final_solver.solve()
        x_out.assign(self.x_out)


class ExplicitTimeDiscretisation(TimeDiscretisation):
    """Base class for explicit time discretisations."""

    def __init__(self, domain, field_name=None, fixed_subcycles=None,
                 subcycle_by_courant=None, solver_parameters=None, limiter=None,
                 options=None):
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
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        super().__init__(domain, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)

        if fixed_subcycles is not None and subcycle_by_courant is not None:
            raise ValueError('Cannot specify both subcycle and subcycle_by '
                             + 'arguments to a time discretisation')
        self.fixed_subcycles = fixed_subcycles
        self.subcycle_by_courant = subcycle_by_courant

    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            apply_bcs (bool, optional): whether boundary conditions are to be
                applied. Defaults to True.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        super().setup(equation, apply_bcs, *active_labels)

        # if user has specified a number of fixed subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
        if self.fixed_subcycles is not None:
            self.dt.assign(self.dt/self.fixed_subcycles)
            self.ncycles = self.fixed_subcycles
        else:
            self.dt = self.dt
            self.ncycles = 1
        self.x0 = Function(self.fs)
        self.x1 = Function(self.fs)

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x_out, self.idx),
            map_if_false=drop)

        return l.form

    @cached_property
    def solver(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs - self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        self.solver_parameters.setdefault('snes_type', 'ksponly')
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    @abstractmethod
    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        pass

    @wrapper_apply
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        # If doing adaptive subcycles, update dt and ncycles here
        if self.subcycle_by_courant is not None:
            self.ncycles = math.ceil(float(self.courant_max)/self.subcycle_by_courant)
            self.dt.assign(self.original_dt/self.ncycles)

        self.x0.assign(x_in)
        for i in range(self.ncycles):
            self.apply_cycle(self.x1, self.x0)
            self.x0.assign(self.x1)
        x_out.assign(self.x1)


class ImplicitMultistage(TimeDiscretisation):
    """
    A class for implementing general diagonally implicit multistage (Runge-Kutta)
    methods based on its Butcher tableau.

    Unlike the explicit method, all upper diagonal a_ij elements are non-zero for implicit methods.

    There are three steps to move from the current solution, y^n, to the new one, y^{n+1}

    For each i = 1, s  in an s stage method
    we have the intermediate solutions:                                       \n
    y_i = y^n +  dt*(a_i1*k_1 + a_i2*k_2 + ... + a_ii*k_i)                    \n
    We compute the gradient at the intermediate location, k_i = F(y_i)        \n

    At the last stage, compute the new solution by:                           \n
    y^{n+1} = y^n + dt*(b_1*k_1 + b_2*k_2 + .... + b_s*k_s)                   \n

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
            butcher_matrix (numpy array): A matrix containing the coefficients of
                a butcher tableau defining a given Runge Kutta time discretisation.
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
            self.xnph = tuple([self.dt*self.butcher_matrix[stage, stage]*a + b for a, b in zip(split(self.x_out), split(self.x1))])
        else:
            self.xnph = self.x1 + self.butcher_matrix[stage, stage]*self.dt*self.x_out
        solver = self.solvers[stage]
        solver.solve()

        self.k[stage].assign(self.x_out)

    def apply(self, x_out, x_in):

        for i in range(self.nStages):
            self.solve_stage(x_in, i)

        x_out.assign(x_in)
        for i in range(self.nStages):
            x_out.assign(x_out + self.butcher_matrix[self.nStages, i]*self.dt*self.k[i])

        if self.limiter is not None:
            self.limiter.apply(x_out)


class ExplicitMultistage(ExplicitTimeDiscretisation):
    """
    A class for implementing general explicit multistage (Runge-Kutta)
    methods based on its Butcher tableau.

    A Butcher tableau is formed in the following way for a s-th order explicit scheme: \n

    All upper diagonal a_ij elements are zero for explicit methods.

    There are three steps to move from the current solution, y^n, to the new one, y^{n+1}

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

    def __init__(self, domain, butcher_matrix, field_name=None, fixed_subcycles=None,
                 subcycle_by_courant=None, solver_parameters=None,
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
        self.nbutcher = int(np.shape(self.butcher_matrix)[0])

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
        self.k = [Function(self.fs) for i in range(self.nStages)]

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(ExplicitMultistage, self).lhs

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
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

    def solve_stage(self, x0, stage):
        self.x1.assign(x0)
        for i in range(stage):
            self.x1.assign(self.x1 + self.dt*Constant(self.butcher_matrix[stage-1, i])*self.k[i])
        for evaluate in self.evaluate_source:
            evaluate(self.x1, self.dt)
        if self.limiter is not None:
            self.limiter.apply(self.x1)
        self.solver.solve()
        self.k[stage].assign(self.x_out)

        if (stage == self.nStages - 1):
            self.x1.assign(x0)
            for i in range(self.nStages):
                self.x1.assign( self.x1 + self.dt*Constant(self.butcher_matrix[stage, i])*self.k[i])
            self.x1.assign(self.x1)

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


class ForwardEuler(ExplicitMultistage):
    """
    Implements the forward Euler timestepping scheme.

    The forward Euler method for operator F is the most simple explicit scheme: \n
    k0 = F[y^n]                                                                 \n
    y^(n+1) = y^n + dt*k0                                                       \n
    """
    def __init__(self, domain, field_name=None, fixed_subcycles=None,
                 subcycle_by_courant=None, solver_parameters=None,
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
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class SSPRK3(ExplicitMultistage):
    u"""
    Implements the 3-stage Strong-Stability-Preserving Runge-Kutta method
    for solving ∂y/∂t = F(y). It can be written as:                           \n

    k0 = F[y^n]                                                               \n
    k1 = F[y^n + dt*k1]                                                       \n
    k2 = F[y^n + (1/4)*dt*(k0+k1)]                                            \n
    y^(n+1) = y^n + (1/6)*dt*(k0 + k1 + 4*k2)                                 \n
    """
    def __init__(self, domain, field_name=None, fixed_subcycles=None,
                 subcycle_by_courant=None, solver_parameters=None,
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
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class RK4(ExplicitMultistage):
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
                 subcycle_by_courant=None, solver_parameters=None,
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
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class Heun(ExplicitMultistage):
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
                 subcycle_by_courant=None, solver_parameters=None,
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
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class BackwardEuler(TimeDiscretisation):
    """
    Implements the backward Euler timestepping scheme.

    The backward Euler method for operator F is the most simple implicit scheme: \n
    y^(n+1) = y^n + dt*F[y^(n+1)].                                               \n
    """
    def __init__(self, domain, field_name=None, solver_parameters=None,
                 limiter=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            fixed_subcycles (int, optional): the number of sub-steps to perform.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods. Defaults to None.
        """
        if not solver_parameters:
            # default solver parameters
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}
        super().__init__(domain=domain, field_name=field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)

    @property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.dt*t)

        return l.form

    @property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x1, old_idx=self.idx),
            map_if_false=drop)

        return r.form

    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        for evaluate in self.evaluate_source:
            evaluate(x_in, self.dt)

        if len(self.evaluate_source) > 0:
            # If we have physics, use x_in as first guess
            self.x_out.assign(x_in)

        self.x1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.x_out)


class ThetaMethod(TimeDiscretisation):
    """
    Implements the theta implicit-explicit timestepping method, which can
    be thought as a generalised trapezium rule.

    The theta implicit-explicit timestepping method for operator F is written as: \n
    y^(n+1) = y^n + dt*(1-theta)*F[y^n] + dt*theta*F[y^(n+1)]                     \n
    for off-centring parameter theta.                                             \n
    """

    def __init__(self, domain, theta, field_name=None,
                 solver_parameters=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            theta (float): the off-centring parameter. theta = 1
                corresponds to a backward Euler method. Defaults to None.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.

        Raises:
            ValueError: if theta is not provided.
        """
        if (theta < 0 or theta > 1):
            raise ValueError("please provide a value for theta between 0 and 1")
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super().__init__(domain, field_name,
                         solver_parameters=solver_parameters,
                         options=options)

        self.theta = theta

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.theta*self.dt*t)

        return l.form

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, old_idx=self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -(1-self.theta)*self.dt*t)

        return r.form

    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        self.x1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.x_out)


class TrapeziumRule(ThetaMethod):
    """
    Implements the trapezium rule timestepping method, also commonly known as
    Crank Nicholson.

    The trapezium rule timestepping method for operator F is written as:      \n
    y^(n+1) = y^n + dt/2*F[y^n] + dt/2*F[y^(n+1)].                            \n
    It is equivalent to the "theta" method with theta = 1/2.                  \n
    """

    def __init__(self, domain, field_name=None, solver_parameters=None,
                 options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        super().__init__(domain, 0.5, field_name,
                         solver_parameters=solver_parameters,
                         options=options)


class MultilevelTimeDiscretisation(TimeDiscretisation):
    """Base class for multi-level timesteppers"""

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
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        super().__init__(domain=domain, field_name=field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)
        self.initial_timesteps = 0

    @abstractproperty
    def nlevels(self):
        pass

    def setup(self, equation, apply_bcs=True, *active_labels):
        super().setup(equation=equation, apply_bcs=apply_bcs, *active_labels)
        for n in range(self.nlevels, 1, -1):
            setattr(self, "xnm%i" % (n-1), Function(self.fs))


class BDF2(MultilevelTimeDiscretisation):
    """
    Implements the implicit multistep BDF2 timestepping method.

    The BDF2 timestepping method for operator F is written as:                \n
    y^(n+1) = (4/3)*y^n - (1/3)*y^(n-1) + (2/3)*dt*F[y^(n+1)]                 \n
    """

    @property
    def nlevels(self):
        return 2

    @property
    def lhs0(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.dt*t)

        return l.form

    @property
    def rhs0(self):
        """Set up the time discretisation's right hand side for inital BDF step."""
        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x1, old_idx=self.idx),
            map_if_false=drop)

        return r.form

    @property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: (2/3)*self.dt*t)

        return l.form

    @property
    def rhs(self):
        """Set up the time discretisation's right hand side for BDF2 steps."""
        xn = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x1, old_idx=self.idx),
            map_if_false=drop)
        xnm1 = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.xnm1, old_idx=self.idx),
            map_if_false=drop)

        r = (4/3.) * xn - (1/3.) * xnm1

        return r.form

    @property
    def solver0(self):
        """Set up the problem and the solver for initial BDF step."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs0-self.rhs0, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"0"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    @property
    def solver(self):
        """Set up the problem and the solver for BDF2 steps."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    def apply(self, x_out, *x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
        """
        if self.initial_timesteps < self.nlevels-1:
            self.initial_timesteps += 1
            solver = self.solver0
        else:
            solver = self.solver

        self.xnm1.assign(x_in[0])
        self.x1.assign(x_in[1])
        solver.solve()
        x_out.assign(self.x_out)


class TR_BDF2(TimeDiscretisation):
    """
    Implements the two stage implicit TR-BDF2 time stepping method, with a
    trapezoidal stage (TR) followed by a second order backwards difference stage (BDF2).

    The TR-BDF2 time stepping method for operator F is written as:                                  \n
    y^(n+g) = y^n + dt*g/2*F[y^n] + dt*g/2*F[y^(n+g)] (TR stage)                                    \n
    y^(n+1) = 1/(g(2-g))*y^(n+g) - (1-g)**2/(g(2-g))*y^(n) + (1-g)/(2-g)*dt*F[y^(n+1)] (BDF2 stage) \n
    for an off-centring parameter g (gamma).                                                        \n
    """
    def __init__(self, domain, gamma, field_name=None,
                 solver_parameters=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            gamma (float): the off-centring parameter
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
        """
        if (gamma < 0. or gamma > 1.):
            raise ValueError("please provide a value for gamma between 0 and 1")
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super().__init__(domain, field_name,
                         solver_parameters=solver_parameters,
                         options=options)

        self.gamma = gamma

    def setup(self, equation, apply_bcs=True, *active_labels):
        super().setup(equation, apply_bcs, *active_labels)
        self.xnpg = Function(self.fs)
        self.xn = Function(self.fs)

    @cached_property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative) for the TR stage."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.xnpg, old_idx=self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: 0.5*self.gamma*self.dt*t)

        return l.form

    @cached_property
    def rhs(self):
        """Set up the time discretisation's right hand side for the TR stage."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.xn, old_idx=self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -0.5*self.gamma*self.dt*t)

        return r.form

    @cached_property
    def lhs_bdf2(self):
        """Set up the discretisation's left hand side (the time derivative) for the BDF2 stage."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: ((1.0-self.gamma)/(2.0-self.gamma))*self.dt*t)

        return l.form

    @cached_property
    def rhs_bdf2(self):
        """Set up the time discretisation's right hand side for the BDF2 stage."""
        xn = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.xn, old_idx=self.idx),
            map_if_false=drop)
        xnpg = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.xnpg, old_idx=self.idx),
            map_if_false=drop)

        r = (1.0/(self.gamma*(2.0-self.gamma)))*xnpg - ((1.0-self.gamma)**2/(self.gamma*(2.0-self.gamma)))*xn

        return r.form

    @cached_property
    def solver_tr(self):
        """Set up the problem and the solver."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.xnpg, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_tr"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    @cached_property
    def solver_bdf2(self):
        """Set up the problem and the solver."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs_bdf2-self.rhs_bdf2, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_bdf2"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
        """
        self.xn.assign(x_in)
        self.solver_tr.solve()
        self.solver_bdf2.solve()
        x_out.assign(self.x_out)


class Leapfrog(MultilevelTimeDiscretisation):
    """
    Implements the multistep Leapfrog timestepping method.

    The Leapfrog timestepping method for operator F is written as:            \n
    y^(n+1) = y^(n-1)  + 2*dt*F[y^n]                                          \n
    """
    @property
    def nlevels(self):
        return 2

    @property
    def rhs0(self):
        """Set up the discretisation's right hand side for initial forward euler step."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, old_idx=self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -self.dt*t)

        return r.form

    @property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(Leapfrog, self).lhs

    @property
    def rhs(self):
        """Set up the discretisation's right hand side for leapfrog steps."""
        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=replace_subject(self.x1, old_idx=self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_true=replace_subject(self.xnm1, old_idx=self.idx),
                        map_if_false=lambda t: -2.0*self.dt*t)

        return r.form

    @property
    def solver0(self):
        """Set up the problem and the solver for initial forward euler step."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs0, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"0"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    @property
    def solver(self):
        """Set up the problem and the solver for leapfrog steps."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    def apply(self, x_out, *x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
        """
        if self.initial_timesteps < self.nlevels-1:
            self.initial_timesteps += 1
            solver = self.solver0
        else:
            solver = self.solver

        self.xnm1.assign(x_in[0])
        self.x1.assign(x_in[1])
        solver.solve()
        x_out.assign(self.x_out)


class AdamsBashforth(MultilevelTimeDiscretisation):
    """
    Implements the explicit multistep Adams-Bashforth timestepping
    method of general order up to 5.

    The general AB timestepping method for operator F is written as:                                      \n
    y^(n+1) = y^n + dt*(b_0*F[y^(n)] + b_1*F[y^(n-1)] + b_2*F[y^(n-2)] + b_3*F[y^(n-3)] + b_4*F[y^(n-4)]) \n
    """
    def __init__(self, domain, order, field_name=None,
                 solver_parameters=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            order (float, optional): order of scheme
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.

        Raises:
            ValueError: if order is not provided, or is in incorrect range.
        """

        if (order > 5 or order < 1):
            raise ValueError("Adams-Bashforth of order greater than 5 not implemented")
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")

        super().__init__(domain, field_name,
                         solver_parameters=solver_parameters,
                         options=options)

        self.order = order

    def setup(self, equation, apply_bcs=True, *active_labels):
        super().setup(equation=equation, apply_bcs=apply_bcs,
                      *active_labels)

        self.x = [Function(self.fs) for i in range(self.nlevels)]

        if (self.order == 1):
            self.b = [1.0]
        elif (self.order == 2):
            self.b = [-(1.0/2.0), (3.0/2.0)]
        elif (self.order == 3):
            self.b = [(5.0)/(12.0), -(16.0)/(12.0), (23.0)/(12.0)]
        elif (self.order == 4):
            self.b = [-(9.0)/(24.0), (37.0)/(24.0), -(59.0)/(24.0), (55.0)/(24.0)]
        elif (self.order == 5):
            self.b = [(251.0)/(720.0), -(1274.0)/(720.0), (2616.0)/(720.0),
                      -(2774.0)/(720.0), (2901.0)/(720.0)]

    @property
    def nlevels(self):
        return self.order

    @property
    def rhs0(self):
        """Set up the discretisation's right hand side for initial forward euler step."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x[-1], old_idx=self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -self.dt*t)

        return r.form

    @property
    def lhs(self):
        """Set up the discretisation's left hand side (the time derivative)."""
        return super(AdamsBashforth, self).lhs

    @property
    def rhs(self):
        """Set up the discretisation's right hand side for Adams Bashforth steps."""
        r = self.residual.label_map(all_terms,
                                    map_if_true=replace_subject(self.x[-1], old_idx=self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -self.b[-1]*self.dt*t)
        for n in range(self.nlevels-1):
            rtemp = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                            map_if_true=drop,
                                            map_if_false=replace_subject(self.x[n], old_idx=self.idx))
            rtemp = rtemp.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: -self.dt*self.b[n]*t)
            r += rtemp
        return r.form

    @property
    def solver0(self):
        """Set up the problem and the solverfor initial forward euler step."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs0, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"0"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    @property
    def solver(self):
        """Set up the problem and the solver for Adams Bashforth steps."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    def apply(self, x_out, *x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
        """
        if self.initial_timesteps < self.nlevels-1:
            self.initial_timesteps += 1
            solver = self.solver0
        else:
            solver = self.solver

        for n in range(self.nlevels):
            self.x[n].assign(x_in[n])
        solver.solve()
        x_out.assign(self.x_out)


class AdamsMoulton(MultilevelTimeDiscretisation):
    """
    Implements the implicit multistep Adams-Moulton
    timestepping method of general order up to 5

    The general AM timestepping method for operator F is written as                      \n
    y^(n+1) = y^n + dt*(b_0*F[y^(n+1)] + b_1*F[y^(n)] + b_2*F[y^(n-1)] + b_3*F[y^(n-2)]) \n
    """
    def __init__(self, domain, order, field_name=None,
                 solver_parameters=None, options=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            order (float, optional): order of scheme
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.

        Raises:
            ValueError: if order is not provided, or is in incorrect range.
        """
        if (order > 4 or order < 1):
            raise ValueError("Adams-Moulton of order greater than 5 not implemented")
        if isinstance(options, (EmbeddedDGOptions, RecoveryOptions)):
            raise NotImplementedError("Only SUPG advection options have been implemented for this time discretisation")
        if not solver_parameters:
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super().__init__(domain, field_name,
                         solver_parameters=solver_parameters,
                         options=options)

        self.order = order

    def setup(self, equation, apply_bcs=True, *active_labels):
        super().setup(equation=equation, apply_bcs=apply_bcs, *active_labels)

        self.x = [Function(self.fs) for i in range(self.nlevels)]

        if (self.order == 1):
            self.bl = (1.0/2.0)
            self.br = [(1.0/2.0)]
        elif (self.order == 2):
            self.bl = (5.0/12.0)
            self.br = [-(1.0/12.0), (8.0/12.0)]
        elif (self.order == 3):
            self.bl = (9.0/24.0)
            self.br = [(1.0/24.0), -(5.0/24.0), (19.0/24.0)]
        elif (self.order == 4):
            self.bl = (251.0/720.0)
            self.br = [-(19.0/720.0), (106.0/720.0), -(254.0/720.0), (646.0/720.0)]

    @property
    def nlevels(self):
        return self.order

    @property
    def rhs0(self):
        """Set up the discretisation's right hand side for initial trapezoidal step."""
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x[-1], old_idx=self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -0.5*self.dt*t)

        return r.form

    @property
    def lhs0(self):
        """Set up the time discretisation's right hand side for initial trapezoidal step."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: 0.5*self.dt*t)
        return l.form

    @property
    def lhs(self):
        """Set up the time discretisation's right hand side for Adams Moulton steps."""
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, old_idx=self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.bl*self.dt*t)
        return l.form

    @property
    def rhs(self):
        """Set up the discretisation's right hand side for Adams Moulton steps."""
        r = self.residual.label_map(all_terms,
                                    map_if_true=replace_subject(self.x[-1], old_idx=self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -self.br[-1]*self.dt*t)
        for n in range(self.nlevels-1):
            rtemp = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                            map_if_true=drop,
                                            map_if_false=replace_subject(self.x[n], old_idx=self.idx))
            rtemp = rtemp.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: -self.dt*self.br[n]*t)
            r += rtemp
        return r.form

    @property
    def solver0(self):
        """Set up the problem and the solver for initial trapezoidal step."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs0-self.rhs0, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"0"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    @property
    def solver(self):
        """Set up the problem and the solver for Adams Moulton steps."""
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    def apply(self, x_out, *x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
        """
        if self.initial_timesteps < self.nlevels-1:
            self.initial_timesteps += 1
            solver = self.solver0
        else:
            solver = self.solver

        for n in range(self.nlevels):
            self.x[n].assign(x_in[n])
        solver.solve()
        x_out.assign(self.x_out)

class SDC(object, metaclass=ABCMeta):

    def __init__(self, domain, M, maxk, quadrature, field_name=None, final_update=True):
        with PETSc.Log.Event("SDC_init"):
            self.field_name=field_name
            self.domain = domain
            self.dt_coarse = domain.dt
            self.M = M
            self.maxk = maxk
            self.final_update=final_update

            self.create_nodes(self.dt_coarse, quadrature)
            self.Qmatrix()
            self.Smatrix()
            self.Qfinal()
            self.dtau = Constant(np.diff(np.append(0, self.nodes)))

    @property
    def nlevels(self):
        return 1

    @abstractmethod
    def setup(self, equation):
        pass

    def create_nodes(self, b, quadrature, A=-1, B=1):
        with PETSc.Log.Event("SDC_nodes"):
            M = self.M
            a = 0.
            nodes = np.zeros(M)
            if quadrature == "gauss-radau":
                # nodes and weights for gauss - radau IIA quadrature
                # See Abramowitz & Stegun p 888
                nodes[0] = A
                p = np.poly1d([1, 1])
                pn = legendre(M)
                pn1 = legendre(M-1)
                poly, remainder = (pn + pn1)/p  # [1] returns remainder from polynomial division
                nodes[1:] = np.sort(poly.roots)
            elif quadrature == "gauss-lobatto":
                pn = legendre(M-1)
                pn_d=pn.deriv()
                nodes[0] = A
                nodes[-1]= B
                nodes[1:-1] = np.sort(pn_d.roots)
            elif quadrature == "gauss-legendre":
                pn = legendre(M)
                nodes = np.sort(pn.roots)

            # rescale to be between [a,b] instead of [A,B]
            nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
            self.nodes = ((b + a) - nodes)[::-1]  # reverse nodes

    def NewtonVM(self, t):
        """
        t: array or list containing nodes.
        returns: array Newton Vandermode Matrix. Entries are in the lower
        triangle
        Polynomial can be created with
        scipy.linalg.solve_triangular(NewtonVM(t),y,lower=True) where y
        contains the points the polynomial need to pass through
        """
        t = np.asarray(t)
        dim = len(t)
        VM = np.zeros([dim, dim])
        VM[:, 0] = 1
        for i in range(1, dim):
            VM[:, i] = (t[:] - t[(i - 1)]) * VM[:, i - 1]

        return VM

    def Horner_newton(self, weights, xi, x):
        """
        Horner scheme to evaluate polynomials based on newton basis
        """
        y = np.zeros_like(x)
        for i in range(len(weights)):
            y = y * (x - xi[(-i - 1)]) + weights[(-i - 1)]

        return y

    def gauss_legendre(self, n, b, A=-1, B=1):
        # nodes and weights for gauss legendre quadrature
        a = 0
        poly = legendre(n)
        polyd = poly.deriv()
        nodes = poly.roots
        nodes = np.sort(nodes)
        weights = 2/((1-nodes**2)*(np.polyval(polyd, nodes))**2)
        gl_nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        gl_weights = (b-a)/(B-A)*weights
        return gl_nodes, gl_weights

    def get_weights(self, b):
        # This calculates for equation 2.4 FWSW - called from Q
        # integrates lagrange polynomials to the points [nodes]
        with PETSc.Log.Event("SDC_weights"):
            M = self.M
            nodes_m, weights_m = self.gauss_legendre(np.ceil(M/2), b)  # use gauss-legendre quadrature to integrate polynomials
            weights = np.zeros(M)
            for j in np.arange(M):
                coeff = np.zeros(M)
                coeff[j] = 1.0  # is unity because it needs to be scaled with y_j for interpolation we have  sum y_j*l_j
                poly_coeffs = scipy.linalg.solve_triangular(self.NewtonVM(self.nodes), coeff, lower=True)
                eval_newt_poly = self.Horner_newton(poly_coeffs, self.nodes, nodes_m)
                weights[j] = np.dot(weights_m, eval_newt_poly)
        return weights

    def Qmatrix(self):
        """
        Integration Matrix
        """
        with PETSc.Log.Event("SDC_Q"):
            M = self.M
            self.Q = np.zeros([M, M])

            # for all nodes, get weights for the interval [tleft,node]
            for m in np.arange(M):
                w = self.get_weights(self.nodes[m])
                self.Q[m, 0:] = w
        
    def Qfinal(self):
        """
        Final Update Integration Vector
        """
        M = self.M
        self.Qfin = np.zeros(M)

        # Get weights for the interval [0,dt]
        w = self.get_weights(self.dt_coarse)
        
        self.Qfin[:] = w

    def Smatrix(self):
        """
        Integration matrix based on Q: sum(S@vector) returns integration
        """
        from copy import deepcopy
        M = self.M
        self.S = np.zeros([M, M])

        self.S[0, :] = deepcopy(self.Q[0, :])
        for m in np.arange(1, M):
            self.S[m, :] = self.Q[m, :] - self.Q[m - 1, :]

    def compute_quad(self):
        with PETSc.Log.Event("SDC_quad"):
            for j in range(self.M):
                self.quad[j].assign(0.)
                for k in range(self.M):
                    self.quad[j] += float(self.S[j, k])*self.fUnodes[k]
    
    def compute_quad_final(self):
        self.quad_final.assign(0.)
        for k in range(self.M):
            self.quad_final += float(self.Qfin[k])*self.fUnodes[k]

    @abstractmethod
    def apply(self, x_out, x_in):
        pass


class FE_SDC(SDC):

    def __init__(self, base_scheme, domain, M, maxk, quadrature, field_name=None, final_update=True):
        super().__init__(domain, M, maxk, quadrature, field_name=field_name, final_update=final_update)
        self.base = base_scheme


    def setup(self, equation, apply_bcs=True, *active_labels):
        self.base.setup(equation, apply_bcs, *active_labels)
        self.residual = self.base.residual

        # set up SDC form and solver
        if self.field_name is not None and hasattr(equation, "field_names"):
            self.idx = equation.field_names.index(self.field_name)
            W = equation.spaces[self.idx]
        else:
            self.field_name = equation.field_name
            W = equation.function_space
            self.idx = None

        self.W = W
        self.Unodes = [Function(W) for _ in range(self.M+1)]
        self.Unodes1 = [Function(W) for _ in range(self.M+1)]
        self.fUnodes = [Function(W) for _ in range(self.M+1)]
        self.quad = [Function(W) for _ in range(self.M+1)]

        self.U_SDC = Function(W)
        self.U0 = Function(W)
        self.Un = Function(W)
        self.Q_ = Function(W)
        self.quad_final = Function(W)
        self.U_fin = Function(W)

        try:
            #bcs = equation.bcs['u']
            self.bcs = [DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain) for bc in equation.bcs['u']]
        except KeyError:
            self.bcs = None

        # set up RHS evaluation
        self.Urhs = Function(W)
        self.Uin = Function(W)

    @property
    def res_rhs(self):
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
        L = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        Frhs = a - L
        return Frhs.form
    
    @property
    def res_SDC(self):
        F = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: self.dt*t)

        a = F.label_map(lambda t: t.has_label(time_derivative),
                        replace_subject(self.U_SDC, old_idx=self.idx),
                        drop)

        F_exp = F.label_map(all_terms, replace_subject(self.Un, old_idx=self.idx))
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)

        F0 = F.label_map(lambda t: t.has_label(time_derivative),
                         drop,
                         replace_subject(self.U0, old_idx=self.idx))
        F0 = F0.label_map(all_terms,
                          lambda t: -1*t)

        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_, old_idx=self.idx),
                                    drop)

        F_SDC = a + F_exp + F0 + Q
        return F_SDC.form

    @property
    def res_fin(self):

        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                        replace_subject(self.U_fin, old_idx=self.idx),
                        drop)

        F_exp = self.residual.label_map(lambda t: t.has_label(time_derivative),
                            replace_subject(self.Un, old_idx=self.idx),
                            drop)
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)


        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.quad_final, old_idx=self.idx),
                                    drop)

        F_fin = a + F_exp + Q
        return F_fin.form
    
    @cached_property
    def solver_fin(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        prob_fin = NonlinearVariationalProblem(self.res_fin, self.U_fin, bcs=self.bcs)
        #solver_name = self.field_name+self.__class__.__name__+"_SDC"
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        #self.solver_parameters.setdefault('snes_type', 'ksponly')
        return NonlinearVariationalSolver(prob_fin)
    
    @cached_property
    def solver_SDC(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        prob_SDC = NonlinearVariationalProblem(self.res_SDC, self.U_SDC, bcs=self.bcs)
        #solver_name = self.field_name+self.__class__.__name__+"_SDC"
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        #self.solver_parameters.setdefault('snes_type', 'ksponly')
        return NonlinearVariationalSolver(prob_SDC)
    
    @cached_property
    def solver_rhs(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        prob_rhs = NonlinearVariationalProblem(self.res_rhs, self.Urhs, bcs=self.bcs)
        #solver_name = self.field_name+self.__class__.__name__+"_rhs"
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        #self.solver_parameters.setdefault('snes_type', 'ksponly')
        return NonlinearVariationalSolver(prob_rhs)

    def apply(self, x_out, x_in):
        self.Un.assign(x_in)

        self.Unodes[0].assign(self.Un)
        for m in range(self.M):
            self.base.dt = float(self.dtau[m])
            self.base.apply(self.Unodes[m+1], self.Unodes[m])

        k = 0
        while k < self.maxk:
            k += 1

            for m in range(1, self.M+1):
                self.Uin.assign(self.Unodes[m])
                self.solver_rhs.solve()
                self.fUnodes[m-1].assign(self.Urhs)

            self.compute_quad()

            self.Unodes1[0].assign(self.Unodes[0])
            for m in range(1, self.M+1):
                self.dt = float(self.dtau[m-1])
                self.U0.assign(self.Unodes[m-1])
                self.Un.assign(self.Unodes1[m-1])
                self.Q_.assign(self.quad[m-1])
                self.solver_SDC.solve(self.Unodes[m-1])
                self.Unodes1[m].assign(self.U_SDC)
            for m in range(1, self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])

            self.Un.assign(self.Unodes1[-1])
        if self.maxk > 0:
            if self.final_update:
                for m in range(1, self.M+1):
                    self.Uin.assign(self.Unodes1[m])
                    self.solver_rhs.solve()
                    self.fUnodes[m-1].assign(self.Urhs)
                self.Un.assign(x_in)
                self.compute_quad_final()
                self.solver_fin.solve()
                x_out.assign(self.U_fin)
            else:
                x_out.assign(self.Unodes1[-1])
        else:
            x_out.assign(self.Unodes[-1])
            


class BE_SDC(SDC):

    def __init__(self, base_scheme, domain, M, maxk, quadrature, field_name=None, final_update=True):
        super().__init__(domain, M, maxk, quadrature, field_name=field_name, final_update=final_update)
        self.base = base_scheme


    def setup(self, equation, apply_bcs=True, *active_labels):
        self.base.setup(equation, *active_labels)
        self.residual = self.base.residual


        # set up SDC form and solver
        if self.field_name is not None and hasattr(equation, "field_names"):
            self.idx = equation.field_names.index(self.field_name)
            W = equation.spaces[self.idx]
        else:
            self.field_name = equation.field_name
            W = equation.function_space
            self.idx = None
        self.W = W
        self.Unodes = [Function(W) for _ in range(self.M+1)]
        self.Unodes1 = [Function(W) for _ in range(self.M+1)]
        self.fUnodes = [Function(W) for _ in range(self.M+1)]
        self.quad = [Function(W) for _ in range(self.M+1)]

        self.U_SDC = Function(W)
        self.U0 = Function(W)
        self.U01 = Function(W)
        self.Un = Function(W)
        self.Q_ = Function(W)
        self.U_fin = Function(W)
        self.quad_final = Function(W)

        try:
            self.bcs = [DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain) for bc in equation.bcs['u']]
        except KeyError:
            self.bcs = None
        
        # set up RHS evaluation
        self.Urhs = Function(W)
        self.Uin = Function(W)
    
    @property
    def res_rhs(self):
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
        L = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        Frhs = a - L
        return Frhs.form
    
    @property
    def res_SDC(self):
        F = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: self.dt*t)

        F_imp = F.label_map(all_terms,
                            replace_subject(self.U_SDC, old_idx=self.idx))

        F_exp = F.label_map(all_terms, replace_subject(self.Un, old_idx=self.idx))
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t,
                                drop)

        F01 = F.label_map(lambda t: t.has_label(time_derivative),
                            drop,
                            replace_subject(self.U01, old_idx=self.idx))

        F01 = F01.label_map(all_terms, lambda t: -1*t)

        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_, old_idx=self.idx),
                                    drop)

        F_SDC = F_imp + F_exp + F01  + Q
        return F_SDC.form
    @property
    def res_fin(self):

        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                        replace_subject(self.U_fin, old_idx=self.idx),
                        drop)

        F_exp = self.residual.label_map(lambda t: t.has_label(time_derivative),
                            replace_subject(self.Un, old_idx=self.idx),
                            drop)
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)


        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.quad_final, old_idx=self.idx),
                                    drop)

        F_fin = a + F_exp + Q
        return F_fin.form
    
    @cached_property
    def solver_fin(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        prob_fin = NonlinearVariationalProblem(self.res_fin, self.U_fin, bcs=self.bcs)
        #solver_name = self.field_name+self.__class__.__name__+"_SDC"
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        #self.solver_parameters.setdefault('snes_type', 'ksponly')
        return NonlinearVariationalSolver(prob_fin)
    
    @cached_property
    def solver_SDC(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        prob_SDC = NonlinearVariationalProblem(self.res_SDC, self.U_SDC, bcs=self.bcs)
        #solver_name = self.field_name+self.__class__.__name__+"_SDC"
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        #self.solver_parameters.setdefault('snes_type', 'ksponly')
        return NonlinearVariationalSolver(prob_SDC)
    
    @cached_property
    def solver_rhs(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        prob_rhs = NonlinearVariationalProblem(self.res_rhs, self.Urhs, bcs=self.bcs)
        #solver_name = self.field_name+self.__class__.__name__+"_rhs"
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        #self.solver_parameters.setdefault('snes_type', 'ksponly')
        return NonlinearVariationalSolver(prob_rhs)

    def apply(self, x_out, x_in):
        self.Un.assign(x_in)

        self.Unodes[0].assign(self.Un)
        for m in range(self.M):
            self.base.dt = self.dtau[m]
            self.base.apply(self.Unodes[m+1], self.Unodes[m])
        k = 0
        while k < self.maxk:
            k += 1

            for m in range(1, self.M+1):
                self.Uin.assign(self.Unodes[m])
                self.solver_rhs.solve()
                self.fUnodes[m-1].assign(self.Urhs)

            self.compute_quad()

            self.Unodes1[0].assign(self.Unodes[0])
            for m in range(1, self.M+1):
                self.dt = float(self.dtau[m-1])
                self.U01.assign(self.Unodes[m])
                self.U0.assign(self.Unodes[m-1])
                self.Un.assign(self.Unodes1[m-1])
                self.Q_.assign(self.quad[m-1])
                self.solver_SDC.solve()
                self.Unodes1[m].assign(self.U_SDC)
            for m in range(1, self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])

            self.Un.assign(self.Unodes1[-1])
        if self.maxk > 0:
            if self.final_update:
                for m in range(1, self.M+1):
                    self.Uin.assign(self.Unodes1[m])
                    self.solver_rhs.solve()
                    self.fUnodes[m-1].assign(self.Urhs)
                self.Un.assign(x_in)
                self.compute_quad_final()
                self.solver_fin.solve()
                x_out.assign(self.U_fin)
            else:
                x_out.assign(self.Unodes1[-1])
        else:
            x_out.assign(self.Unodes[-1])

class IMEX_SDC(SDC):

    def __init__(self, base_scheme, domain, M, maxk, quadrature, field_name=None,final_update=True):
        super().__init__(domain, M, maxk, quadrature, field_name=field_name,final_update=final_update)
        self.base = base_scheme

    def setup(self, equation, apply_bcs=True, *active_labels):
        with PETSc.Log.Event("IMEX_SDC_init"):
            self.base.setup(equation, *active_labels)
            self.residual = self.base.residual
            self.ksp_tols=[1e-3,1e-2,1e-1,1e-1]
            self.snes_tols=[1e-4,1e-3,1e-2,1e-1]

            # set up SDC form and solver
            if self.field_name is not None and hasattr(equation, "field_names"):
                self.idx = equation.field_names.index(self.field_name)
                W = equation.spaces[self.idx]
            else:
                self.field_name = equation.field_name
                W = equation.function_space
                self.idx = None

            # set up SDC form and solver
            self.W = W
            self.Unodes = [Function(W) for _ in range(self.M+1)]
            self.Unodes1 = [Function(W) for _ in range(self.M+1)]
            self.fUnodes = [Function(W) for _ in range(self.M+1)]
            self.quad = [Function(W) for _ in range(self.M+1)]

            self.U_SDC = Function(W)
            self.U0 = Function(W)
            self.U01 = Function(W)
            self.Un = Function(W)
            self.Q_ = Function(W)
            self.U_fin = Function(W)
            self.quad_final = Function(W)

            try:
                #bcs = equation.bcs['u']
                self.bcs = [DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain) for bc in equation.bcs['u']]
            except KeyError:
                self.bcs = None
            # set up RHS evaluation
            self.Urhs = Function(W)
            self.Uin = Function(W)

    @property
    def res_rhs(self):
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
        L = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        Frhs = a - L
        return Frhs.form
    
    @property
    def res_SDC(self):
        F = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: self.dt*t)

        F_imp = F.label_map(lambda t: any(t.has_label(time_derivative, implicit)),
                            replace_subject(self.U_SDC, old_idx=self.idx),
                            drop)

        F_exp = F.label_map(lambda t: any(t.has_label(time_derivative, explicit)),
                            replace_subject(self.Un, old_idx=self.idx),
                            drop)
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)

        F01 = F.label_map(lambda t: t.has_label(implicit),
                          replace_subject(self.U01, old_idx=self.idx),
                          drop)

        F01 = F01.label_map(all_terms, lambda t: -1*t)

        F0 = F.label_map(lambda t: t.has_label(explicit),
                         replace_subject(self.U0, old_idx=self.idx),
                         drop)
        F0 = F0.label_map(all_terms, lambda t: -1*t)

        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_, old_idx=self.idx),
                                    drop)

        F_SDC = F_imp + F_exp + F01 + F0 + Q
        return F_SDC.form
    
    @property
    def res_fin(self):

        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                        replace_subject(self.U_fin, old_idx=self.idx),
                        drop)

        F_exp = self.residual.label_map(lambda t: t.has_label(time_derivative),
                            replace_subject(self.Un, old_idx=self.idx),
                            drop)
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)


        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.quad_final, old_idx=self.idx),
                                    drop)

        F_fin = a + F_exp + Q
        return F_fin.form
    
    @cached_property
    def solver_fin(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        prob_fin = NonlinearVariationalProblem(self.res_fin, self.U_fin, bcs=self.bcs)
        
        #solver_name = self.field_name+self.__class__.__name__+"_SDC"
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        solver_parameters = {'snes_type': 'ksponly',
                             'ksp_type': 'cg',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}
        return NonlinearVariationalSolver(prob_fin, solver_parameters=solver_parameters)
    
    @cached_property
    def solver_SDC(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        prob_SDC = NonlinearVariationalProblem(self.res_SDC, self.U_SDC, bcs=self.bcs)
        #solver_name = self.field_name+self.__class__.__name__+"_SDC"
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        solver_parameters = {'snes_type': 'newtonls',
                             'ksp_type': 'gmres',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}
        # solver_parameters= { 'snes_type': 'newtonls',
        #                      'ksp_type': 'gmres', 
        #                      "pc_type": "mg",
        #                     "pc_mg_type": "full",
        #                     "mg_levels_ksp_type": "chebyshev",
        #                     "mg_levels_ksp_max_it": 2,
        #                     "mg_levels_pc_type": "jacobi"
        #                     }
        #solver_parameters.setdefault('snes_type', 'ksponly')
        return NonlinearVariationalSolver(prob_SDC, solver_parameters=solver_parameters)
    
    @cached_property
    def solver_rhs(self):
        """Set up the problem and the solver."""
        # setup linear solver using lhs and rhs defined in derived class
        prob_rhs = NonlinearVariationalProblem(self.res_rhs, self.Urhs, bcs=self.bcs)
        #solver_name = self.field_name+self.__class__.__name__+"_rhs"
        # If snes_type not specified by user, set this to ksp only to avoid outer Newton iteration
        solver_parameters = {'snes_type': 'ksponly',
                             'ksp_type': 'cg',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}
        return NonlinearVariationalSolver(prob_rhs, solver_parameters=solver_parameters)

    def apply(self, x_out, x_in):
        self.Un.assign(x_in)

        self.Unodes[0].assign(self.Un)
        with PETSc.Log.Event("IMEX_SDC_precon"):
            for m in range(self.M):
                self.Unodes[m+1].assign(self.Un)
                # self.base.dt = float(self.dtau[m])
                # self.base.apply(self.Unodes[m+1], self.Unodes[m])

        k = 0
        while k < self.maxk:
            k += 1
            with PETSc.Log.Event("IMEX_SDC_rhs_quad"):
                for m in range(1, self.M+1):
                    self.Uin.assign(self.Unodes[m])
                    self.solver_rhs.solve()
                    self.fUnodes[m-1].assign(self.Urhs)
                self.compute_quad()

            self.Unodes1[0].assign(self.Unodes[0])
            for m in range(1, self.M+1):
                self.dt = float(self.dtau[m-1])
                self.U0.assign(self.Unodes[m-1])
                self.U01.assign(self.Unodes[m])
                self.Un.assign(self.Unodes1[m-1])
                self.Q_.assign(self.quad[m-1])
                with PETSc.Log.Event("IMEX_SDC_solve"):
                    self.U_SDC.assign(self.Unodes[m])
                    # self.solver_SDC.parameters['ksp_rtol']=self.ksp_tols[k]
                    # self.solver_SDC.parameters['snes_rtol']=self.snes_tols[k]
                    # self.solver_SDC.parameters['ksp_atol']=self.ksp_tols[k]
                    # self.solver_SDC.parameters['snes_atol']=self.snes_tols[k]
                    self.solver_SDC.solve()
                    self.Unodes1[m].assign(self.U_SDC)
            for m in range(1, self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])

            self.Un.assign(self.Unodes1[-1])
        if self.maxk > 0:
            if self.final_update:
                for m in range(1, self.M+1):
                    self.Uin.assign(self.Unodes1[m])
                    with PETSc.Log.Event("IMEX_SDC_rhs2"):
                        self.solver_rhs.solve()
                    self.fUnodes[m-1].assign(self.Urhs)
                self.Un.assign(x_in)
                self.compute_quad_final()
                with PETSc.Log.Event("IMEX_SDC_final_solve"):
                    self.solver_fin.solve()
                x_out.assign(self.U_fin)
            else:
                x_out.assign(self.Unodes1[-1])
        else:
            x_out.assign(self.Unodes[-1])

class ImplicitMidpoint(ImplicitMultistage):
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


class QinZhang(ImplicitMultistage):
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


class IMEX_Euler(IMEXMultistage):
    u"""
    Implements IMEX Euler one-stage method.

    The method, for solving                                                    \n
    ∂y/∂t = F(y) + S(y), can be written as:                                    \n

    y_0 = y^n                                                                  \n
    y_1 = y^n + dt*F[y_1] + dt*S[y_0]                                          \n
    y^(n+1) = y^n + dt*F[y_1] + dt*S[y_0]                                      \n
    """
    def __init__(self, domain, field_name=None, solver_parameters=None, limiter=None, options=None):
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
        butcher_imp = np.array([[0., 0.], [0., 1.], [0., 1.]])
        butcher_exp = np.array([[0., 0.], [1., 0.], [1., 0.]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class ARS3(IMEXMultistage):
    u"""
    Implements ARS3(2,3,3) two-stage IMEX Runge–Kutta method
    from RK IMEX for HEVI (Weller et al 2013).
    Where g = (3 + sqrt(3))/6.

    The method, for solving                                                    \n
    ∂y/∂t = F(y) + S(y), can be written as:                                    \n

    y_0 = y^n                                                                  \n
    y_1 = y^n + dt*g*F[y_1] + dt*g*S[y_0]                                      \n
    y_2 = y^n + dt*((1-2g)*F[y_1]+g*F[y_2])                                    \n
              + dt*((g-1)*S[y_0]+2(g-1)*S[y_1])                                \n
    y^(n+1) = y^n + dt*(g*F[y_1]+(1-g)*F[y_2])                                 \n
                  + dt*(0.5*S[y_1]+0.5*S[y_2])                                 \n
    """
    def __init__(self, domain, field_name=None, solver_parameters=None, limiter=None, options=None):
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
        g = (3. + np.sqrt(3.))/6.
        butcher_imp = np.array([[0., 0., 0.], [0., g, 0.], [0., 1-2.*g, g], [0., 0.5, 0.5]])
        butcher_exp = np.array([[0., 0., 0.], [g, 0., 0.], [g-1., 2.*(1.-g), 0.], [0., 0.5, 0.5]])

        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class ARK2(IMEXMultistage):
    u"""
    Implements ARK2(2,3,2) two-stage IMEX Runge–Kutta method from
    RK IMEX for HEVI (Weller et al 2013).
    Where g = 1 - 1/sqrt(2), a = 1/6(3 + 2sqrt(2)), d = 1/2sqrt(2).

    The method, for solving                                                    \n
    ∂y/∂t = F(y) + S(y), can be written as:                                    \n

    y_0 = y^n                                                                  \n
    y_1 = y^n + dt*(g*F[y_0]+g*F[y_1]) + 2*dt*g*S[y_0]                         \n
    y_2 = y^n + dt*(d*F[y_0]+d*F[y_1]+g*F[y_2])                                \n
              + dt*((1-a)*S[y_0]+a*S[y_1])                                     \n
    y^(n+1) = y^n + dt*(d*F[y_0]+d*F[y_1]+g*F[y_2])                            \n
                  + dt*(d*S[y_0]+d*S[y_1]+g*S[y_2])                            \n
    """
    def __init__(self, domain, field_name=None, solver_parameters=None, limiter=None, options=None):
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
        g = 1. - 1./np.sqrt(2.)
        d = 1./(2.*np.sqrt(2.))
        a = 1./6.*(3. + 2.*np.sqrt(2.))
        butcher_imp = np.array([[0., 0., 0.], [g, g, 0.], [d, d, g], [d, d, g]])
        butcher_exp = np.array([[0., 0., 0.], [2.*g, 0., 0.], [1.-a, a, 0.], [d, d, g]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class SSP3(IMEXMultistage):
    u"""
    Implements SSP3(3,3,2) three-stage IMEX Runge–Kutta method from RK IMEX for HEVI (Weller et al 2013).
    Where g = 1 - 1/sqrt(2)

    The method, for solving                                                    \n
    ∂y/∂t = F(y) + S(y), can be written as:                                    \n

    y_1 = y^n + dt*g*F[y_1]                                                    \n
    y_2 = y^n + dt*((1-2g)*F[y_1]+g*F[y_2]) + dt*S[y_1]                        \n
    y_3 = y^n + dt*((0.5-g)*F[y_1]+g*F[y_3]) + dt*(0.25*S[y_1]+0.25*S[y_2])    \n
    y^(n+1) = y^n + dt*(1/6*F[y_1]+1/6*F[y_2]+2/3*F[y_3])                      \n
                  + dt*(1/6*S[y_1]+1/6*S[y_2]+2/3*S[y_3])                      \n
    """
    def __init__(self, domain, field_name=None, solver_parameters=None, limiter=None, options=None):
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
        g = 1. - (1./np.sqrt(2.))
        butcher_imp = np.array([[g, 0., 0.], [1-2.*g, g, 0.], [0.5-g, 0., g], [(1./6.), (1./6.), (2./3.)]])
        butcher_exp = np.array([[0., 0., 0.], [1., 0., 0.], [0.25, 0.25, 0.], [(1./6.), (1./6.), (2./3.)]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)


class Trap2(IMEXMultistage):
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
    def __init__(self, domain, field_name=None, solver_parameters=None, limiter=None, options=None):
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
        e = 0.
        butcher_imp = np.array([[0., 0., 0., 0.], [e, 0., 0., 0.], [0.5, 0., 0.5, 0.], [0.5, 0., 0., 0.5], [0.5, 0., 0., 0.5]])
        butcher_exp = np.array([[0., 0., 0., 0.], [1., 0., 0., 0.], [0.5, 0.5, 0., 0.], [0.5, 0., 0.5, 0.], [0.5, 0., 0.5, 0.]])
        super().__init__(domain, butcher_imp, butcher_exp, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)
