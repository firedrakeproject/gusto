"""
This module defines the SpatialMethod base object, which is used to define a
spatial discretisation of some term.
"""

from firedrake import split
from firedrake.fml import Term, keep, drop, all_terms
from gusto.core.labels import prognostic

__all__ = ['SpatialMethod']


class SpatialMethod(object):
    """
    The base object for describing a spatial discretisation of some term.
    """

    def __init__(self, equation, variable, *term_labels):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                the original type of this term.
            variable (str): name of the variable to set the method for
            term_labels (:class:`Label`): One or more labels specifying which type(s)
                of terms should be discretized.
        """
        self.equation = equation
        self.variable = variable
        self.domain = self.equation.domain
        self.term_labels = list(term_labels)

        if hasattr(equation, "field_names"):
            # Equation with multiple prognostic variables
            variable_idx = equation.field_names.index(variable)
            self.test = equation.tests[variable_idx]
            self.field = split(equation.X)[variable_idx]
        else:
            self.field = equation.X
            self.test = equation.test

        if (len(self.term_labels) == 1):
            # Most cases only have one term to be replaced
            self.term_label = self.term_labels[0]
            self.original_form = equation.residual.label_map(
                lambda t: t.has_label(self.term_label) and t.get(prognostic) == variable,
                map_if_true=keep,
                map_if_false=drop
            )
            # Check that the original form has the correct number of terms
            num_terms = len(self.original_form.terms)
            assert num_terms == 1, f'Unable to find {self.term_label.label} term ' \
                + f'for {variable}. {num_terms} found'
        else:
            # Multiple terms to be replaced. Find the original terms to be used
            self.term_label = self.term_labels[0]
            self.original_form = equation.residual.label_map(
                all_terms,
                map_if_true=drop
            )
            for term in self.term_labels:
                original_form = equation.residual.label_map(
                    lambda t: t.has_label(term) and t.get(prognostic) == variable,
                    map_if_true=keep,
                    map_if_false=drop
                )
                # Check that the original form has the correct number of terms
                num_terms = len(original_form.terms)
                assert num_terms == 1, f'Unable to find {term.label} term ' \
                    + f'for {variable}. {num_terms} found'
                # Add the terms form to original forms
                self.original_form += original_form

    def replace_form(self, equation):
        """
        Replaces the form for the term in the equation with the form for the
        specific discretisation.

        Args:
            equation (:class:`PrognosticEquation`): the equation or scheme whose
                term should be replaced with the specific term of this
                discretisation.
        """

        # Replace original term with new term
        equation.residual = equation.residual.label_map(
            lambda t: t.has_label(self.term_label) and t.get(prognostic) == self.variable,
            map_if_true=lambda t: Term(self.form.form, t.labels))
