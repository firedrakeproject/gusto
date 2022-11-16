"""Provides forms for describing diffusion terms."""

from firedrake import (inner, outer, grad, avg, dx, dS_h, dS_v, dS,
                       FacetNormal)
from gusto.labels import diffusion


__all__ = ["interior_penalty_diffusion_form"]


def interior_penalty_diffusion_form(state, test, q, parameters):
    u"""
    Form for the interior penalty discretisation of a diffusion term, ∇.(κ∇q)

    The interior penalty discretisation involves the factor 'mu', the penalty
    weight function.

    Args:
        state (:class:`State`): the model's state object.
        test (:class:`TestFunction`): the equation's test function.
        q (:class:`Function`): the variable being diffused.
        parameters (:class:`DiffusionParameters`): object containing metadata
            describing the diffusion term. Includes kappa and mu.

    Returns:
        :class:`ufl.Form`: the diffusion form.
    """

    dS_ = (dS_v + dS_h) if state.mesh.extruded else dS
    kappa = parameters.kappa
    mu = parameters.mu

    n = FacetNormal(state.mesh)

    form = inner(grad(test), grad(q)*kappa)*dx

    def get_flux_form(dS, M):
        """
        The facet term for the interior penalty diffusion discretisation.

        Args:
            dS (:class:`ufl.Measure`): the facet measure.
            M (:class:`Constant`): the diffusivity.

        Returns:
            :class:`ufl.Form`: the interior penalty flux form
        """

        fluxes = (
            -inner(2*avg(outer(q, n)), avg(grad(test)*M))
            - inner(avg(grad(q)*M), 2*avg(outer(test, n)))
            + mu*inner(2*avg(outer(q, n)), 2*avg(outer(test, n)*kappa))
        )*dS
        return fluxes

    form += get_flux_form(dS_, kappa)

    return diffusion(form)
