"""
Defines objects to perform parametrisations of physical processes, or "physics".

"PhysicsParametrisation" schemes are routines to compute updates to prognostic
fields that represent the action of non-fluid processes, or those fluid
processes that are unresolved. This module contains a set of these processes in
the form of classes with "evaluate" methods.
"""

from abc import ABCMeta, abstractmethod
from firedrake import Function, dx, Projector, assemble, split
from firedrake.__future__ import interpolate
from firedrake.fml import subject
from gusto.core.labels import PhysicsLabel, source_label
from gusto.core.logging import logger


__all__ = ["PhysicsParametrisation", "SourceSink"]


class PhysicsParametrisation(object, metaclass=ABCMeta):
    """Base class for the parametrisation of physical processes for Gusto."""

    def __init__(self, equation, label_name, parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            label_name (str): name of physics scheme, to be passed to its label.
            parameters (:class:`Configuration`, optional): parameters containing
                the values of gas constants. Defaults to None, in which case the
                parameters are obtained from the equation.
        """

        self.label = PhysicsLabel(label_name)
        self.equation = equation
        if parameters is None and hasattr(equation, 'parameters'):
            self.parameters = equation.parameters
        else:
            self.parameters = parameters

    @abstractmethod
    def evaluate(self):
        """
        Computes the value of physics source and sink terms.
        """
        pass


class SourceSink(PhysicsParametrisation):
    """
    The source or sink of some variable, described through a UFL expression.

    A scheme representing the general source or sink of a variable, described
    through a UFL expression. The expression should be for the rate of change
    of the variable. It is intended that the source/sink is independent of the
    prognostic variables.

    The expression can also be a time-varying expression. In which case a
    function should be provided, taking a :class:`Constant` as an argument (to
    represent the time.)
    """

    def __init__(self, equation, variable_name, rate_expression,
                 time_varying=False, method='interpolate'):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            variable_name (str): the name of the variable
            rate_expression (:class:`ufl.Expr` or func): an expression giving
                the rate of change of the variable. If a time-varying expression
                is needed, this should be a function taking a single argument
                representing the time. Then the `time_varying` argument must
                be set to True.
            time_varying (bool, optional): whether the source/sink expression
                varies with time. Defaults to False.
            method (str, optional): the method to use to evaluate the expression
                for the source. Valid options are 'interpolate' or 'project'.
                Default is 'interpolate'.
        """

        label_name = f'source_sink_{variable_name}'
        super().__init__(equation, label_name, parameters=None)

        if method not in ['interpolate', 'project']:
            raise ValueError(f'Method {method} for source/sink evaluation not valid')
        self.method = method
        self.time_varying = time_varying
        self.variable_name = variable_name

        # Check the variable exists
        if hasattr(equation, "field_names"):
            assert variable_name in equation.field_names, \
                f'Field {variable_name} does not exist in the equation set'
        else:
            assert variable_name == equation.field_name, \
                f'Field {variable_name} does not exist in the equation'

        # Work out the appropriate function space
        if hasattr(equation, "field_names"):
            self.V_idx = equation.field_names.index(variable_name)
            W = equation.function_space
            V = W.sub(self.V_idx)
            test = equation.tests[self.V_idx]
            self.V_idx = self.V_idx
            self.source = Function(W)
            self.source_expr = split(self.source)[self.V_idx]
            self.source_int = self.source.subfunctions[self.V_idx]
        else:
            V = equation.function_space
            test = equation.test
            self.source = Function(V)
            self.source_expr = self.source
            self.source_int = self.source

        # Make source/sink term
        equation.residual += source_label(
            self.label(subject(test * self.source_expr * dx, self.source), self.evaluate)
        )

        # Handle whether the expression is time-varying or not
        if self.time_varying:
            expression = rate_expression(equation.domain.t)
        else:
            expression = rate_expression

        # Handle method of evaluating source/sink
        if self.method == 'interpolate':
            self.source_interpolate = interpolate(expression, self.source_int)
        else:
            self.source_projector = Projector(expression, self.source_int)

        # If not time-varying, evaluate for the first time here
        if not self.time_varying:
            if self.method == 'interpolate':
                self.source_int.assign(assemble(self.source_interpolate))
            else:
                self.source_projector.project()

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evalutes the source term generated by the physics.

        Args:
            x_in: (:class:`Function`): the (mixed) field to be evolved. Unused.
            dt: (:class:`Constant`): the timestep, which can be the time
                interval for the scheme. Unused.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """
        if self.time_varying:
            logger.info(f'Evaluating physics parametrisation {self.label.label}')
            if self.method == 'interpolate':
                self.source_int.assign(assemble(self.source_interpolate))
            else:
                self.source_projector.project()
        else:
            pass
        # If a source output is provided, assign the source term to it
        if x_out is not None:
            x_out.assign(self.source)
