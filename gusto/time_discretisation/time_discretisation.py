u"""
Objects for discretising time derivatives.

Time discretisation objects discretise ∂y/∂t = F(y), for variable y, time t and
operator F.
"""

from abc import ABCMeta, abstractmethod
import math

from firedrake import (Function, TestFunction, TestFunctions, DirichletBC,
                       Constant, NonlinearVariationalProblem,
                       NonlinearVariationalSolver)
from firedrake.fml import (replace_subject, replace_test_function, Term,
                           all_terms, drop, subject)
from firedrake.formmanipulation import split_form
from firedrake.utils import cached_property

from gusto.core.configuration import EmbeddedDGOptions, RecoveryOptions
from gusto.core.labels import (time_derivative, prognostic, physics_label,
                               mass_weighted, nonlinear_time_derivative, source_label)
from gusto.core.logging import logger, DEBUG, logging_ksp_monitor_true_residual
from gusto.time_discretisation.wrappers import *
from gusto.solvers import mass_parameters

__all__ = ["TimeDiscretisation", "ExplicitTimeDiscretisation", "BackwardEuler",
           "ThetaMethod", "TrapeziumRule", "TR_BDF2"]


def wrapper_apply(original_apply):
    """Decorator to add steps for using a wrapper around the apply method."""
    def get_apply(self, x_out, x_in):

        if self.augmentation is not None:

            def new_apply(self, x_out, x_in):

                self.augmentation.pre_apply(x_in)
                original_apply(self, self.augmentation.x_out, self.augmentation.x_in)
                self.augmentation.post_apply(x_out)

            return new_apply(self, x_out, x_in)

        elif self.wrapper is not None:

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

    def __init__(self, domain, field_name=None, subcycling_options=None,
                 solver_parameters=None, limiter=None, options=None,
                 augmentation=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
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
        self.augmentation = augmentation
        self.subcycling_options = subcycling_options

        if self.subcycling_options is not None:
            self.subcycling_options.check_options()

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
                            'Time discretisation: suboption SUPG is not implemented within MixedOptions')
                    else:
                        raise RuntimeError(
                            f'Time discretisation: suboption wrapper {suboption.name} not implemented')

            elif self.wrapper_name == "embedded_dg":
                self.wrapper = EmbeddedDGWrapper(self, options)
            elif self.wrapper_name == "recovered":
                self.wrapper = RecoveryWrapper(self, options)
            elif self.wrapper_name == "supg":
                self.suboptions = options.suboptions
                self.wrapper = SUPGWrapper(self, options)
            else:
                raise RuntimeError(
                    f'Time discretisation: wrapper {self.wrapper_name} not implemented')
        else:
            self.wrapper = None
            self.wrapper_name = None

        # get default solver options if none passed in
        if solver_parameters is None:
            self.solver_parameters = {'ksp_type': 'gmres',
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
            if isinstance(self.field_name, list):
                # Multiple fields are being solved for simultaneously.
                # This enables conservative transport to be implemented with SIQN.
                # Use the full mixed space for self.fs, with the
                # field_name, residual, and BCs being set up later.
                self.fs = equation.function_space
                self.idx = None
            else:
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

        if self.augmentation is not None:
            self.fs = self.augmentation.fs
            self.residual = self.augmentation.residual
            self.idx = None

        if len(active_labels) > 0:
            if isinstance(self.field_name, list):
                # Multiple fields are being solved for simultaneously.
                # Keep all time derivative terms:
                residual = self.residual.label_map(
                    lambda t: t.has_label(time_derivative),
                    map_if_false=drop)

                # Only keep active labels for prognostics in the list
                # of simultaneously transported variables:
                for subname in self.field_name:
                    field_residual = self.residual.label_map(
                        lambda t: t.get(prognostic) == subname,
                        map_if_false=drop)

                    residual += field_residual.label_map(
                        lambda t: t.has_label(*active_labels),
                        map_if_false=drop)

                self.residual = residual

            else:
                self.residual = self.residual.label_map(
                    lambda t: any(t.has_label(time_derivative, *active_labels)),
                    map_if_false=drop)

        # Set the field name if using simultaneous transport.
        if isinstance(self.field_name, list):
            self.field_name = equation.field_name

        if self.augmentation is not None:
            # Transfer BCs from appropriate function space
            bcs = self.augmentation.bcs if hasattr(self.augmentation, "bcs") else None
        else:
            bcs = equation.bcs[self.field_name]

        self.evaluate_source = []
        self.physics_names = []
        for t in self.residual:
            if t.has_label(physics_label):
                physics_name = t.get(physics_label)
                if t.labels[physics_name] not in self.physics_names:
                    self.evaluate_source.append(t.labels[physics_name])
                    self.physics_names.append(t.labels[physics_name])

        # Check if there are any mass-weighted terms:
        if len(self.residual.label_map(lambda t: t.has_label(mass_weighted), map_if_false=drop)) > 0:
            if self.augmentation is not None:
                if self.augmentation.name == 'mean_mixing_ratio':
                    field_names = self.augmentation.field_names
            else:
                field_names = equation.field_names

            for field in field_names:
                # Check if the mass term for this prognostic is mass-weighted
                if len(self.residual.label_map((
                    lambda t: t.get(prognostic) == field
                    and t.has_label(time_derivative)
                    and t.has_label(mass_weighted)
                ), map_if_false=drop)) == 1:

                    field_terms = self.residual.label_map(
                        lambda t: t.get(prognostic) == field and not t.has_label(time_derivative),
                        map_if_false=drop
                    )

                    # Check that the equation for this prognostic does not involve
                    # both mass-weighted and non-mass-weighted terms; if so, a split
                    # timestepper should be used instead.
                    if len(field_terms.label_map(lambda t: t.has_label(mass_weighted), map_if_false=drop)) > 0:
                        if len(field_terms.label_map(lambda t: not t.has_label(mass_weighted), map_if_false=drop)) > 0:
                            raise ValueError('Mass-weighted and non-mass-weighted terms are present in a '
                                             + f'timestepping equation for {field}. As these terms cannot '
                                             + 'be solved for simultaneously, a split timestepping method '
                                             + 'should be used instead.')
                        else:
                            # Replace the terms with a mass_weighted label with the
                            # mass_weighted form. It is important that the labels from
                            # this new form are used.
                            self.residual = self.residual.label_map(
                                lambda t: t.get(prognostic) == field and t.has_label(mass_weighted),
                                map_if_true=lambda t: t.get(mass_weighted))
                            print('mass-weighted stuff has been replaced')


        # -------------------------------------------------------------------- #
        # Set up Wrappers
        # -------------------------------------------------------------------- #

        if self.wrapper is not None:

            wrapper_bcs = bcs if apply_bcs else None

            if self.wrapper_name == "mixed_options":

                self.wrapper.wrapper_spaces = equation.spaces
                self.wrapper.field_names = equation.field_names

                for field, subwrapper in self.wrapper.subwrappers.items():

                    if field not in equation.field_names:
                        raise ValueError(f'The option defined for {field} is for a field '
                                         + 'that does not exist in the equation set.')

                    field_idx = equation.field_names.index(field)
                    subwrapper.setup(equation.spaces[field_idx], equation.bcs[field])

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
                    if self.suboptions is not None:
                        for field_name, term_labels in self.suboptions.items():
                            self.wrapper.setup(field_name)
                            new_test = self.wrapper.test
                            if term_labels is not None:
                                for term_label in term_labels:
                                    self.residual = self.residual.label_map(
                                        lambda t: t.get(prognostic) == field_name and t.has_label(term_label),
                                        map_if_true=replace_test_function(new_test, old_idx=self.wrapper.idx))
                            else:
                                self.residual = self.residual.label_map(
                                    lambda t: t.get(prognostic) == field_name,
                                    map_if_true=replace_test_function(new_test, old_idx=self.wrapper.idx))
                            self.residual = self.wrapper.label_terms(self.residual)
                    else:
                        self.wrapper.setup(self.field_name)
                        new_test = self.wrapper.test
                        self.residual = self.residual.label_map(
                            all_terms,
                            map_if_true=replace_test_function(new_test))
                        self.residual = self.wrapper.label_terms(self.residual)
                else:
                    self.wrapper.setup(self.fs, wrapper_bcs)
                    self.fs = self.wrapper.function_space
                    new_test = TestFunction(self.wrapper.test_space)
                    # Replace the original test function with the one from the wrapper
                    self.residual = self.residual.label_map(
                        all_terms,
                        map_if_true=replace_test_function(new_test))

                    self.residual = self.wrapper.label_terms(self.residual)
                if self.solver_parameters is None:
                    self.solver_parameters = self.wrapper.solver_parameters

        # -------------------------------------------------------------------- #
        # Make boundary conditions
        # -------------------------------------------------------------------- #

        if not apply_bcs:
            self.bcs = None
        elif self.wrapper is not None and self.wrapper_name != "supg":
            if self.wrapper_name == 'mixed_options':
                # Define new Dirichlet BCs on the wrapper-modified
                # mixed function space.
                self.bcs = []
                for idx, field_name in enumerate(self.equation.field_names):
                    for bc in equation.bcs[field_name]:
                        self.bcs.append(DirichletBC(self.fs.sub(idx),
                                                    bc.function_arg,
                                                    bc.sub_domain))
            else:
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

        for term in self.residual:
            print('\n')
            print(term.get(subject))
            print(term.form)
            print(term.labels)
        print(len(self.residual))

       # import sys; sys.exit()


    @property
    def nlevels(self):
        return 1

    @property
    def res(self):
        """Set up the discretisation's residual."""
        residual = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x_out, old_idx=self.idx),
            map_if_false=drop
        )
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, old_idx=self.idx)
        )

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: -self.dt * t
        )

        residual -= r

        return residual.form

    @cached_property
    def solver(self):
        """Set up the problem and the solver."""
        # setup solver using residual (res) defined in derived class
        problem = NonlinearVariationalProblem(self.res, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        solver = NonlinearVariationalSolver(
            problem,
            solver_parameters=self.solver_parameters,
            options_prefix=solver_name
        )
        if logger.isEnabledFor(DEBUG):
            solver.snes.ksp.setMonitor(logging_ksp_monitor_true_residual)
        return solver

    def update_subcycling(self):
        """
        Update the time step and number of substeps when adaptively subcycling.
        """

        if (self.subcycling_options is not None
                and self.subcycling_options.subcycle_by_courant is not None):

            subcycle_by_courant = self.subcycling_options.subcycle_by_courant
            max_subcycles = self.subcycling_options.max_subcycles

            # Set number of subcycles
            self.ncycles = math.ceil(float(self.courant_max)/subcycle_by_courant)

            # Cap number of subcycles
            if max_subcycles is not None:
                if self.ncycles > max_subcycles:
                    logger.warning(
                        'Adaptive subcycling: capping number of subcycles at '
                        f'{max_subcycles}'
                    )
                    self.ncycles = max_subcycles

            logger.debug(f'Performing {self.ncycles} subcycles')
            self.dt.assign(self.original_dt/self.ncycles)

    @abstractmethod
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        pass


class ExplicitTimeDiscretisation(TimeDiscretisation):
    """Base class for explicit time discretisations."""

    def __init__(self, domain, field_name=None, subcycling_options=None,
                 solver_parameters=None, limiter=None, options=None,
                 augmentation=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
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
        super().__init__(domain, field_name,
                         subcycling_options=subcycling_options,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options,
                         augmentation=augmentation)

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

        # get default solver options if none passed in
        self.solver_parameters.update(mass_parameters(
            self.fs, equation.domain.spaces))
        self.solver_parameters['snes_type'] = 'ksponly'

        # if user has specified a number of fixed subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
        if (self.subcycling_options is not None
                and self.subcycling_options.fixed_subcycles is not None):
            self.ncycles = self.subcycling_options.fixed_subcycles
            self.dt.assign(self.dt/self.ncycles)
        else:
            self.ncycles = 1
            self.dt = self.dt
        self.x0 = Function(self.fs)
        self.x1 = Function(self.fs)

        # If the time_derivative term is nonlinear, we must use a nonlinear solver,
        # but if the time_derivative term is linear, we can reuse the factorisations.
        if (
            len(self.residual.label_map(
                lambda t: t.has_label(nonlinear_time_derivative),
                map_if_false=drop
            )) > 0 and self.solver_parameters.get('snes_type') == 'ksponly'
        ):
            message = ('Switching to newton line search'
                       + f' nonlinear solver for {self.field_name}'
                       + ' as the time derivative term is nonlinear')
            logger.warning(message)
            self.solver_parameters['snes_type'] = 'newtonls'
        else:
            self.solver_parameters.setdefault('snes_lag_jacobian', -2)
            self.solver_parameters.setdefault('snes_lag_jacobian_persists', None)
            self.solver_parameters.setdefault('snes_lag_preconditioner', -2)
            self.solver_parameters.setdefault('snes_lag_preconditioner_persists', None)

    @cached_property
    def res(self):
        """Set up the discretisation's residual"""
        residual = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x_out, old_idx=self.idx),
            map_if_false=drop
        )

        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, old_idx=self.idx)
        )

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: -self.dt * t
        )

        residual -= r

        return residual.form

    @cached_property
    def solver(self):
        """Set up the problem and the solver."""
        # setup linear solver using residual (res) defined in derived class
        problem = NonlinearVariationalProblem(self.res, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
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
        self.update_subcycling()

        self.x0.assign(x_in)
        for i in range(self.ncycles):
            self.subcycle_idx = i
            self.apply_cycle(self.x1, self.x0)
            self.x0.assign(self.x1)
        x_out.assign(self.x1)


class BackwardEuler(TimeDiscretisation):
    """
    Implements the backward Euler timestepping scheme.

    The backward Euler method for operator F is the most simple implicit scheme: \n
    y^(n+1) = y^n + dt*F[y^(n+1)].                                               \n
    """
    def __init__(self, domain, field_name=None, subcycling_options=None,
                 solver_parameters=None, limiter=None, options=None,
                 augmentation=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
            """
        if not solver_parameters:
            # default solver parameters
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}
        super().__init__(domain=domain, field_name=field_name,
                         subcycling_options=subcycling_options,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options,
                         augmentation=augmentation)

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
        if (self.subcycling_options is not None
                and self.subcycling_options.fixed_subcycles is not None):
            self.dt.assign(self.dt/self.fixed_subcycles)
            self.ncycles = self.fixed_subcycles
        else:
            self.dt = self.dt
            self.ncycles = 1
        self.x0 = Function(self.fs)
        self.x1 = Function(self.fs)

    @property
    def res(self):
        """Set up the discretisation's residual."""
        residual = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(
                self.x_out, old_idx=self.idx
            )
        )
        residual = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: self.dt*t
        )
        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.x1, old_idx=self.idx),
            map_if_false=drop
        )

        residual -= r

        return residual.form

    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation through a single sub-step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """

        for evaluate in self.evaluate_source:
            evaluate(x_in, self.dt)

        self.x1.assign(x_in)
        # Set initial solver guess
        self.x_out.assign(x_in)
        self.solver.solve()
        x_out.assign(self.x_out)

    @wrapper_apply
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        self.update_subcycling()

        self.x0.assign(x_in)
        for i in range(self.ncycles):
            self.subcycle_idx = i
            self.apply_cycle(self.x1, self.x0)
            self.x0.assign(self.x1)
        x_out.assign(self.x1)


class ThetaMethod(TimeDiscretisation):
    """
    Implements the theta implicit-explicit timestepping method, which can
    be thought as a generalised trapezium rule.

    The theta implicit-explicit timestepping method for operator F is written
    as:                                                                       \n
    y^(n+1) = y^n + dt*(1-theta)*F[y^n] + dt*theta*F[y^(n+1)]                 \n
    for off-centring parameter theta.
    """

    def __init__(self, domain, theta, field_name=None, subcycling_options=None,
                 solver_parameters=None, options=None, augmentation=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            theta (float): the off-centring parameter. theta = 1
                corresponds to a backward Euler method. Defaults to None.
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            subcycling_options(:class:`SubcyclingOptions`, optional): an object
                containing options for subcycling the time discretisation.
                Defaults to None.
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying solver. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.

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
                         subcycling_options=subcycling_options,
                         solver_parameters=solver_parameters,
                         options=options,
                         augmentation=augmentation)

        self.theta = theta

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
        if (self.subcycling_options is not None
                and self.subcycling_options.fixed_subcycles is not None):
            self.dt.assign(self.dt/self.fixed_subcycles)
            self.ncycles = self.fixed_subcycles
        else:
            self.dt = self.dt
            self.ncycles = 1
        self.x0 = Function(self.fs)
        self.x1 = Function(self.fs)

    @cached_property
    def res(self):
        """Set up the discretisation's residual."""
        residual = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, old_idx=self.idx)
        )
        residual = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: self.theta * self.dt * t
        )

        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x1, old_idx=self.idx)
        )
        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: -(1 - self.theta) * self.dt * t
        )
        residual -= r
        return residual.form

    def apply_cycle(self, x_out, x_in):
        """
        Apply the time discretisation for a single substep.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        for evaluate in self.evaluate_source:
            evaluate(x_in, self.dt)

        self.x1.assign(x_in)
        # Set initial solver guess
        self.x_out.assign(x_in)
        self.solver.solve()
        x_out.assign(self.x_out)

    @wrapper_apply
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        self.update_subcycling()
        if self.augmentation is not None:
            self.augmentation.update(x_in)

        self.x0.assign(x_in)
        for i in range(self.ncycles):
            self.subcycle_idx = i
            self.apply_cycle(self.x1, self.x0)
            self.x0.assign(self.x1)
        x_out.assign(self.x1)


class TrapeziumRule(ThetaMethod):
    """
    Implements the trapezium rule timestepping method, also commonly known as
    Crank Nicholson.

    The trapezium rule timestepping method for operator F is written as:      \n
    y^(n+1) = y^n + dt/2*F[y^n] + dt/2*F[y^(n+1)].                            \n
    It is equivalent to the "theta" method with theta = 1/2.                  \n
    """

    def __init__(self, domain, field_name=None, subcycling_options=None,
                 solver_parameters=None, options=None, augmentation=None):
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
            augmentation (:class:`Augmentation`): allows the equation solved in
                this time discretisation to be augmented, for instances with
                extra terms of another auxiliary variable. Defaults to None.
        """
        super().__init__(domain, 0.5, field_name,
                         subcycling_options=subcycling_options,
                         solver_parameters=solver_parameters,
                         options=options, augmentation=augmentation)


class TR_BDF2(TimeDiscretisation):
    """
    Implements the two stage implicit TR-BDF2 time stepping method, with a
    trapezoidal stage (TR) followed by a second order backwards difference stage
    (BDF2).

    The TR-BDF2 time stepping method for operator F is written as:            \n
    y^(n+g) = y^n + dt*g/2*F[y^n] + dt*g/2*F[y^(n+g)] (TR stage)              \n
    y^(n+1) = 1/(g(2-g))*y^(n+g) - (1-g)**2/(g(2-g))*y^(n) + (1-g)/(2-g)*dt*F[y^(n+1)] (BDF2 stage) \n
    for an off-centring parameter g (gamma).
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
    def res(self):
        """Set up the discretisation's residual for the TR stage."""
        residual = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.xnpg, old_idx=self.idx)
        )
        residual = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: 0.5 * self.gamma * self.dt * t
        )

        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.xn, old_idx=self.idx)
        )
        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: -0.5 * self.gamma * self.dt * t
        )
        residual -= r

        return residual.form

    @cached_property
    def res_bdf2(self):
        """Set up the discretisation's residual for the BDF2 stage."""
        residual = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.x_out, old_idx=self.idx)
        )
        residual = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: (1.0 - self.gamma) / (2.0 - self.gamma) * self.dt * t
        )

        xn = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.xn, old_idx=self.idx),
            map_if_false=drop
        )
        xnpg = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.xnpg, old_idx=self.idx),
            map_if_false=drop
        )

        r = (1.0 / (self.gamma * (2.0 - self.gamma))) * xnpg - \
            ((1.0 - self.gamma) ** 2 / (self.gamma * (2.0 - self.gamma))) * xn

        residual -= r
        return residual.form

    @cached_property
    def solver_tr(self):
        """Set up the problem and the solver."""
        # setup solver using residual (res) defined in derived class
        problem = NonlinearVariationalProblem(self.res, self.xnpg, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_tr"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    @cached_property
    def solver_bdf2(self):
        """Set up the problem and the solver."""
        # setup solver using residual (res) defined in derived class
        problem = NonlinearVariationalProblem(self.res_bdf2, self.x_out, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_bdf2"
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters,
                                          options_prefix=solver_name)

    @wrapper_apply
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field(s).
        """
        self.xn.assign(x_in)

        # Set initial solver guess
        self.xnpg.assign(x_in)
        self.solver_tr.solve()

        # Set initial solver guess
        self.x_out.assign(self.xnpg)
        self.solver_bdf2.solve()
        x_out.assign(self.x_out)
