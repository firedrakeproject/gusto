"""Provides discretisations for diffusion terms."""

from firedrake import inner, outer, grad, avg, dx, dS_h, dS_v, dS, FacetNormal
from gusto.labels import diffusion
from gusto.spatial_methods import SpatialMethod


__all__ = ["InteriorPenaltyDiffusion"]


class DiffusionMethod(SpatialMethod):
    """
    The base object for describing a spatial discretisation of diffusion terms.
    """

    def __init__(self, equation, variable):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                a diffusion term.
            variable (str): name of the variable to set the diffusion scheme for
        """

        # Inherited init method extracts original term to be replaced
        super().__init__(equation, variable, diffusion)


def interior_penalty_diffusion_form(domain, test, q, parameters):
    u"""
    Form for the interior penalty discretisation of a diffusion term, ∇.(κ∇q)

    The interior penalty discretisation involves the factor 'mu', the penalty
    weight function.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the equation's test function.
        q (:class:`Function`): the variable being diffused.
        parameters (:class:`DiffusionParameters`): object containing metadata
            describing the diffusion term. Includes kappa and mu.

    Returns:
        :class:`ufl.Form`: the diffusion form.
    """

    dS_ = (dS_v + dS_h) if domain.mesh.extruded else dS
    kappa = parameters.kappa
    mu = parameters.mu

    n = FacetNormal(domain.mesh)

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


def interior_penalty_diffusion_form_1d(domain, test, q, parameters):
    u"""
    Form for the interior penalty discretisation of a diffusion term, ∇.(κ∇q)

    The interior penalty discretisation involves the factor 'mu', the penalty
    weight function.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the equation's test function.
        q (:class:`Function`): the variable being diffused.
        parameters (:class:`DiffusionParameters`): object containing metadata
            describing the diffusion term. Includes kappa and mu.

    Returns:
        :class:`ufl.Form`: the diffusion form.
    """

    kappa = parameters.kappa
    mu = parameters.mu

    n = FacetNormal(domain.mesh)[0]

    form = test.dx(0) * q.dx(0) * kappa * dx

    def get_flux_form(M):
        """
        The facet term for the interior penalty diffusion discretisation.

        Args:
            dS (:class:`ufl.Measure`): the facet measure.
            M (:class:`Constant`): the diffusivity.

        Returns:
            :class:`ufl.Form`: the interior penalty flux form
        """

        fluxes = (
            -2*avg(q * n) * avg(test.dx(0) * M)
            -avg(q.dx(0) * M) * 2 * avg(test * n)
            + mu * 2 * avg(q * n) * 2 * avg(test * n)*kappa
        )*dS
        return fluxes

    form += get_flux_form(kappa)

    return diffusion(form)


class InteriorPenaltyDiffusion(DiffusionMethod):
    """The interior penalty method for discretising the diffusion term."""

    def __init__(self, equation, variable, diffusion_parameters):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                a transport term.
            variable (str): name of the variable to set the diffusion method for
            diffusion_parameters (:class:`DiffusionParameters`): object
                containing metadata describing the diffusion term. Includes
                the kappa and mu constants.
        """

        super().__init__(equation, variable)

        if equation.domain.mesh.topological_dimension() == 1:
            self.form = interior_penalty_diffusion_form_1d(
                equation.domain, self.test, self.field, diffusion_parameters)
        else:
            self.form = interior_penalty_diffusion_form(
                equation.domain, self.test, self.field, diffusion_parameters)
