from firedrake import (TestFunction, TrialFunction, Function,
                       inner, outer, grad, avg, dx, dS_h, dS_v,
                       FacetNormal, LinearVariationalProblem,
                       LinearVariationalSolver)
from gusto.form_manipulation_labelling import (drop,
                                               subject, time_derivative,
                                               diffusion, replace_labelled)


__all__ = ["Diffusion", "interior_penalty_diffusion_form"]


class Diffusion(object):
    """
    Base class for diffusion schemes for gusto.

    :arg state: :class:`.State` object.
    """

    def __init__(self, state, fieldname, equation):

        dt = state.dt
        field = state.fields(fieldname)
        trial = TrialFunction(field.function_space())
        self.phi = Function(field.function_space())
        self.phi1 = Function(field.function_space())

        a = equation().label_map(lambda t: t.has_label(time_derivative), replace_labelled("subject", trial), drop)
        a += dt*equation().label_map(lambda t: t.has_label(diffusion), replace_labelled("subject", trial), drop)

        L = equation().label_map(lambda t: t.has_label(time_derivative), replace_labelled("subject", self.phi), drop)

        problem = LinearVariationalProblem(a.form, L.form, self.phi1)
        self.solver = LinearVariationalSolver(problem)

    def apply(self, x, x_out):
        """
        Function takes x as input, computes F(x) and returns x_out
        as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        self.phi.assign(x)
        self.solver.solve()
        x_out.assign(self.phi1)


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

    form = subject(inner(grad(gamma), grad(phi)*kappa)*dx, phi)

    def get_flux_form(dS, M):

        fluxes = (-inner(2*avg(outer(phi, n)), avg(grad(gamma)*M))
                  - inner(avg(grad(phi)*M), 2*avg(outer(gamma, n)))
                  + mu*inner(2*avg(outer(phi, n)), 2*avg(outer(gamma, n)*kappa)))*dS
        return fluxes

    form += subject(get_flux_form(dS_v, kappa), phi)
    form += subject(get_flux_form(dS_h, kappa), phi)

    return diffusion(form)
