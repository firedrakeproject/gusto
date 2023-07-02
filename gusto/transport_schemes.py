"""
Defines TransportScheme objects, which are used to solve a transport problem.
"""

from firedrake import split
from gusto.configuration import IntegrateByParts, TransportEquationType
from gusto.fml import Term, keep, drop, Label, LabelledForm
from gusto.labels import transporting_velocity, prognostic, transport, subject
from gusto.transport_forms import *
import ufl

__all__ = ["DGUpwind", "SUPGTransport"]

# TODO: settle on name: discretisation vs scheme

class TransportScheme(object):
    """
    The base object for describing a transport scheme.
    """

    def __init__(self, equation, variable):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                a transport term.
            variable (str): name of the variable to set the transport scheme for
        """
        self.equation = equation
        self.variable = variable
        self.domain = self.equation.domain

        # TODO: how do we deal with plain transport equation?
        variable_idx = equation.field_names.index(variable)
        self.test = equation.tests[variable_idx]
        self.field = split(equation.X)[variable_idx]

        # Find the original transport term to be used
        # TODO: do we need this?
        self.original_form = equation.residual.label_map(
            lambda t: t.has_label(transport) and t.get(prognostic) == variable,
            map_if_true=keep, map_if_false=drop)

        num_terms = len(self.original_form.terms)
        assert num_terms == 1, \
            f'Unable to find transport term for {variable}. {num_terms} found'

    def replace_transport_form(self, equation):
        """
        Replaces the form for the transport term in the equation with the
        form for the transport discretisation.

        Args:
            equation (:class:`PrognosticEquation`): the equation or scheme whose
                transport term should be replaced with the transport term of
                this discretisation.
        """

        # Add the form to the equation
        equation.residual = equation.residual.label_map(
            lambda t: t.has_label(transport) and t.get(prognostic) == self.variable,
            map_if_true=lambda t: Term(self.form.form, t.labels))


    def setup(self, uadv, equation):
        """
        Set up the transport scheme by replacing the transporting velocity used
        in the form.

        Args:
            uadv (:class:`ufl.Expr`, optional): the transporting velocity.
                Defaults to None.
            equation (:class:`PrognosticEquation`): the equation or scheme whose
                transport term should be replaced with the transport term of
                this discretisation.
        """

        assert self.form.terms[0].has_label(transporting_velocity), \
            'Cannot set up transport scheme on a term that has no transporting velocity'

        if uadv == "prognostic":
            # Find prognostic wind field
            uadv = split(self.original_form.terms[0].get(subject))[0]

        import pdb; pdb.set_trace()

        equation.residual = equation.residual.label_map(
            lambda t: t.has_label(transporting_velocity),
            map_if_true=lambda t:
            Term(ufl.replace(t.form, {t.get(transporting_velocity): uadv}), t.labels)
        )

        import pdb; pdb.set_trace()

        equation.residual = transporting_velocity.update_value(equation.residual, uadv)

        import pdb; pdb.set_trace()

class DGUpwind(TransportScheme):
    """
    The Discontinuous Galerkin Upwind transport scheme.

    Discretises the gradient of a field weakly, taking the upwind value of the
    transported variable at facets.
    """
    def __init__(self, equation, variable, ibp=IntegrateByParts.ONCE,
                 vector_manifold_correction=False, outflow=False):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                a transport term.
            variable (str): name of the variable to set the transport scheme for
            ibp (:class:`IntegrateByParts`, optional): an enumerator for how
                many times to integrate by parts. Defaults to `ONCE`.
            vector_manifold_correction (bool, optional): whether to include a
                vector manifold correction term. Defaults to False.
            outflow (bool, optional): whether to include outflow at the domain
                boundaries, through exterior facet terms. Defaults to False.
        """

        super().__init__(equation, variable)
        self.ibp = ibp
        self.vector_manifold_correction = vector_manifold_correction
        self.outflow = outflow

        # -------------------------------------------------------------------- #
        # Determine appropriate form to use
        # -------------------------------------------------------------------- #

        if self.original_form.terms[0].get(transport) == TransportEquationType.advective:
            if vector_manifold_correction:
                form = vector_manifold_advection_form(self.domain, self.test,
                                                      self.field, ibp=ibp,
                                                      outflow=outflow)
            else:
                form = advection_form(self.domain, self.test, self.field,
                                     ibp=ibp, outflow=outflow)

        elif self.original_form.terms[0].get(transport) == TransportEquationType.conservative:
            if vector_manifold_correction:
                form = vector_manifold_continuity_form(self.domain, self.test,
                                                       self.field, ibp=ibp,
                                                       outflow=outflow)
            else:
                form = continuity_form(self.domain, self.test, self.field,
                                       ibp=ibp, outflow=outflow)

        elif self.original_form.terms[0].get(transport) == TransportEquationType.vector_invariant:
            if outflow:
                raise NotImplementedError('Outflow not implemented for upwind vector invariant')
            form = vector_invariant_form(self.domain, self.test, self.field, ibp=ibp)

        else:
            raise NotImplementedError('Upwind transport scheme has not been '
                                      + 'implemented for this transport equation type')

        self.form = form


class SUPGTransport(TransportScheme):
    pass