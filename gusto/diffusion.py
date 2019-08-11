from abc import ABCMeta, abstractmethod
from firedrake import (TestFunction, TrialFunction, Function,
                       inner, outer, grad, avg, dx, dS_h, dS_v,
                       FacetNormal, LinearVariationalProblem,
                       LinearVariationalSolver, action, Interpolator,
                       Projector, FunctionSpace, as_vector)
from gusto.recovery import Recoverer


__all__ = ["InteriorPenalty", "RecoveredDiffusion"]


class Diffusion(object, metaclass=ABCMeta):
    """
    Base class for diffusion schemes for gusto.

    :arg state: :class:`.State` object.
    """

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


class InteriorPenalty(Diffusion):
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

    def __init__(self, state, V, kappa, mu, bcs=None):
        super(InteriorPenalty, self).__init__(state)

        dt = state.timestepping.dt
        bcs = self.bcs
        gamma = TestFunction(V)
        phi = TrialFunction(V)
        self.phi1 = Function(V)
        n = FacetNormal(state.mesh)
        a = inner(gamma, phi)*dx + dt*inner(grad(gamma), grad(phi)*kappa)*dx

        def get_flux_form(dS, M):

            fluxes = (-inner(2*avg(outer(phi, n)), avg(grad(gamma)*M))
                      - inner(avg(grad(phi)*M), 2*avg(outer(gamma, n)))
                      + mu*inner(2*avg(outer(phi, n)), 2*avg(outer(gamma, n)*kappa)))*dS
            return fluxes

        a += dt*get_flux_form(dS_v, kappa)
        a += dt*get_flux_form(dS_h, kappa)
        L = inner(gamma, phi)*dx
        problem = LinearVariationalProblem(a, action(L, self.phi1), self.phi1, bcs=bcs)
        self.solver = LinearVariationalSolver(problem)

    def apply(self, x_in, x_out):
        self.phi1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.phi1)


class RecoveredDiffusion(Diffusion):
    """
    A diffusion strategy for the lowest order spaces, allowing us to recover to a
    higher order space, where diffusion will be performed.

    :arg diffusion_scheme: an existing Diffusion class. For a vector field this could be a list.
    :arg recovered_options: a RecoveredOptions object.
    """

    def __init__(self, state, diffusion_scheme, V0, recovered_options, vector=False):
        super(RecoveredDiffusion, self).__init__(state)

        DG1 = FunctionSpace(state.mesh, "DG", 1)

        self.diffusion = diffusion_scheme
        bcs = self.diffusion.bcs
        self.vector = vector
        dim = recovered_options.recovered_space.value_size

        if self.vector and len(self.diffusion) != dim:
            raise ValueError('If you want to use a vector method, you must supply a list of diffusion schemes.')

        self.x_in = Function(V0)
        x_rec = Function(recovered_options.recovered_space)
        x_brok = Function(recovered_options.broken_space)

        if self.vector:
            self.x_dg_list = [Function(DG1) for i in range(dim)]
            self.xdg_interpolator_list = [Interpolator(self.x_in[i] + x_rec[i] - x_brok[i], self.x_dg_list[i]) for i in range(dim)]
            self.project_back = Projector(as_vector(self.x_dg_list), self.x_in, bcs=bcs)
        else:
            self.x_dg = Function(recovered_options.embedding_space)
            self.xdg_interpolator = Interpolator(self.x_in + x_rec - x_brok, self.x_dg)
            self.project_back = Projector(self.x_dg, self.x_in, bcs=project_back_bcs)

        self.x_rec_projector = Recoverer(self.x_in, x_rec, VDG=recovered_options.embedding_space, boundary_method=recovered_options.boundary_method)
        self.x_brok_projector = Projector(x_rec, x_brok)

    def apply(self, x_in, x_out):
        self.x_in.assign(x_in)
        self.x_rec_projector.project()
        self.x_brok_projector.project()
        if self.vector:
            for i, (xdg_interpolator, diffusion_scheme) in enumerate(zip(self.xdg_interpolator_list, self.diffusion)):
                xdg_interpolator.interpolate()
                diffusion_scheme.apply(self.x_dg_list[i], self.x_dg_list[i])
        else:
            self.xdg_interpolator.interpolate()
            self.diffusion.apply(self.x_dg, self.x_dg)
        self.project_back.project()
        x_out.assign(self.x_in)
