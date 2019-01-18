from firedrake import (TestFunction, Function,
                       inner, outer, grad, avg, dx, dS_h, dS_v,
                       FacetNormal)
from gusto.form_manipulation_labelling import subject, diffusion


__all__ = ["interior_penalty_diffusion_form"]


def interior_penalty_diffusion_form(state, V, *, kappa, mu):
    """
    Interior penalty diffusion method

    :arg state: :class:`.State` object.
    :arg V: Function space of diffused field
    :arg direction: list containing directions in which function space
    :arg: mu: the penalty weighting function, which is
    :recommended to be proportional to 1/dx
    :arg: kappa: strength of diffusion
    :arg: bcs: (optional) a list of boundary conditions to apply

    """

    gamma = TestFunction(V)
    phi = Function(V)
    n = FacetNormal(state.mesh)

    form = subject(-inner(grad(gamma), grad(phi)*kappa)*dx, phi)

    def get_flux_form(dS, M):

        fluxes = (inner(2*avg(outer(phi, n)), avg(grad(gamma)*M))
                  + inner(avg(grad(phi)*M), 2*avg(outer(gamma, n)))
                  - mu*inner(2*avg(outer(phi, n)), 2*avg(outer(gamma, n)*kappa)))*dS
        return fluxes

    form += subject(get_flux_form(dS_v, kappa), phi)
    form += subject(get_flux_form(dS_h, kappa), phi)

    return diffusion(form)
