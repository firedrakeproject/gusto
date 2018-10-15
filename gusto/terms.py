from abc import ABCMeta, abstractmethod
from firedrake import FacetNormal, div, dx, inner


class Term(object, metaclass=ABCMeta):

    off_centering = 0.5

    def __init__(self, state, test):
        self.state = state
        self.test = test
        self.parameters = state.parameters
        self.n = FacetNormal(state.mesh)

    @abstractmethod
    def evaluate(self, q, fields):
        pass

    def __call__(self, q, fields):
        return self.evaluate(q, fields)


class ShallowWaterPressureGradientTerm(Term):

    def evaluate(self, q, fields):
        g = self.parameters.g
        D = fields("D")
        return g*div(self.test)*D*dx


class ShallowWaterCoriolisTerm(Term):

    def evaluate(self, q, fields):
        f = self.state.fields("coriolis")
        return -f*inner(self.test, self.state.perp(q))*dx


class ShallowWaterTopographyTerm(Term):

    def evaluate(self, q, fields):
        g = self.parameters.g
        b = self.state.fields("topography")
        return g*div(self.test)*b*dx
