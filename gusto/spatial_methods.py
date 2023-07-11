"""
This module defines the SpatialMethod base object, which is used to define a
spatial discretisation of some term.
"""

from firedrake import split
from gusto.fml import Term, keep, drop
from gusto.labels import prognostic

__all__ = ['SpatialMethod']


class SpatialMethod(object):
    """
    The base object for describing a spatial discretisation of some term.
    """

    def __init__(self, equation, variable, term_label):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                the original type of this term.
            variable (str): name of the variable to set the transport scheme for
            term_label (:class:`Label`): the label specifying which type of term
                to be discretised.
        """
        self.equation = equation
        self.variable = variable
        self.domain = self.equation.domain
        self.term_label = term_label

        if hasattr(equation, "field_names"):
            # Equation with multiple prognostic variables
            variable_idx = equation.field_names.index(variable)
            self.test = equation.tests[variable_idx]
            self.field = split(equation.X)[variable_idx]
        else:
            self.field = equation.X
            self.test = equation.test

        # Find the original transport term to be used, which we use to extract
        # information about the transport equation type
        self.original_form = equation.residual.label_map(
            lambda t: t.has_label(term_label) and t.get(prognostic) == variable,
            map_if_true=keep, map_if_false=drop)

        num_terms = len(self.original_form.terms)
        assert num_terms == 1, f'Unable to find {term_label.label} term ' \
            + f'for {variable}. {num_terms} found'

    def replace_form(self, equation):
        """
        Replaces the form for the transport term in the equation with the
        form for the transport discretisation.

        Args:
            equation (:class:`PrognosticEquation`): the equation or scheme whose
                transport term should be replaced with the transport term of
                this discretisation.
        """

        # Replace original term with new term
        equation.residual = equation.residual.label_map(
            lambda t: t.has_label(self.term_label) and t.get(prognostic) == self.variable,
            map_if_true=lambda t: Term(self.form.form, t.labels))
