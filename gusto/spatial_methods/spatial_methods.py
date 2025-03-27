"""
This module defines the SpatialMethod base object, which is used to define a
spatial discretisation of some term.
"""

from firedrake import split
from firedrake.fml import Term, keep, drop
from gusto.core.labels import prognostic

__all__ = ['SpatialMethod']


class SpatialMethod(object):
    """
    The base object for describing a spatial discretisation of some term.
    """

    def __init__(self, equation, variable, term_labels):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                the original type of this term.
            variable (str): name of the variable to set the method for
            term_labels (list of :class:`Label`): list of the labels specifying
                which type of term to be discretised.
        """
        self.equation = equation
        self.variable = variable
        self.domain = self.equation.domain
        # Most schemes have only one term label
        self.term_label = term_labels[0]

        if hasattr(equation, "field_names"):
            # Equation with multiple prognostic variables
            variable_idx = equation.field_names.index(variable)
            self.test = equation.tests[variable_idx]
            self.field = split(equation.X)[variable_idx]
        else:
            self.field = equation.X
            self.test = equation.test

        # Find the original term to be used (for first term label)
        self.original_form = equation.residual.label_map(
            lambda t: t.has_label(self.term_label) and t.get(prognostic) == variable,
            map_if_true=keep, map_if_false=drop)

        num_terms_per_label = len(self.original_form.terms) // len(term_labels)
        assert len(self.original_form.terms) % len(term_labels) == 0, (
            "The terms do not divide evenly into labels."
        )
        assert num_terms_per_label == 1, (
            f"Unable to find terms {[term.label for term in term_labels]} for "
            f"{variable}. {num_terms_per_label} terms per expected term found"
        )

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
