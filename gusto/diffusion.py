from firedrake import (inner, outer, grad, avg, dx, dS_h, dS_v, dS,
                       FacetNormal)
from gusto.labels import diffusion


__all__ = ["interior_penalty_diffusion_form"]


def interior_penalty_diffusion_form(state, test, q, parameters):
    """
    Interior penalty diffusion form

    :arg state: :class:`.State` object.
    :arg V: Function space of diffused field
    :arg direction: list containing directions in which function space
    :arg: mu: the penalty weighting function, which is
    :recommended to be proportional to 1/dx
    :arg: kappa: strength of diffusion

    """

    dS_ = (dS_v + dS_h) if q.function_space().extruded else dS
    kappa = parameters.kappa
    mu = parameters.mu

    n = FacetNormal(state.mesh)

    form = inner(grad(test), grad(q)*kappa)*dx

    def get_flux_form(dS, M):

        fluxes = (
            -inner(2*avg(outer(q, n)), avg(grad(test)*M))
            - inner(avg(grad(q)*M), 2*avg(outer(test, n)))
            + mu*inner(2*avg(outer(q, n)), 2*avg(outer(test, n)*kappa))
        )*dS
        return fluxes

    form += get_flux_form(dS_, kappa)

    return diffusion(form)
