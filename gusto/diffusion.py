from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import TestFunction, TrialFunction, \
    Function, inner, outer, grad, avg, dx, dS_h, dS_v, \
    FacetNormal, LinearVariationalProblem, LinearVariationalSolver, action


class Diffusion(object):
    """
    Base class for diffusion schemes for gusto.

    :arg state: :class:`.State` object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        self.state = state

    @abstractmethod
    def apply(self, x, x_out):
        """
        Function takes x as input, computes F(x) and returns x_out
        as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass


class InteriorPenulty(Diffusion):
    """
    Interior penulty diffusion method

    :arg state: :class:`.State` object.
    :arg V: Function space of diffused field
    :arg direction: list containing directions in which function space
    is discontinuous: 1 corresponds to vertical, 2 to horizontal.
    :arg params: dictionary containing the interior penulty parameters
    :mu and kappa where mu is the penulty weighting function, which is
    :recommended to be proportional to 1/dx

    """

    def __init__(self, state, V, direction=[1,2], params=None):
        super(InteriorPenulty, self).__init__(state)

        dt = state.timestepping.dt
        kappa = params['kappa']
        mu = params['mu']
        gamma = TestFunction(V)
        phi = TrialFunction(V)
        self.phi1 = Function(V)
        n = FacetNormal(state.mesh)
        a = inner(gamma,phi)*dx + dt*inner(grad(gamma), grad(phi)*kappa)*dx

        def get_flux_form(dS, M):

            fluxes = (-inner(2*avg(outer(phi, n)), avg(grad(gamma)*M))
                      - inner(avg(grad(phi)*M), 2*avg(outer(gamma, n)))
                      + mu*inner(2*avg(outer(phi, n)), 2*avg(outer(gamma, n)*kappa)))*dS
            return fluxes

        if 1 in direction:
            a += dt*get_flux_form(dS_v, kappa)
        if 2 in direction:
            a += dt*get_flux_form(dS_h, kappa)
        L = inner(gamma,phi)*dx
        problem = LinearVariationalProblem(a, action(L,self.phi1), self.phi1)
        self.solver = LinearVariationalSolver(problem)

    def apply(self, x_in, x_out):
        self.phi1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.phi1)
