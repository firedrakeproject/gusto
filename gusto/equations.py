from abc import ABCMeta, abstractproperty
from collections import OrderedDict
from firedrake import TestFunction, TrialFunction, MixedFunctionSpace, \
    FiniteElement
from gusto.diagnostics import Diagnostics
from gusto.state import FieldCreator
from gusto.terms import *
from gusto.transport_terms import *


class Equation(object):
    """
    Base equation class.

    Creates test and trial functions on the given function space,
    defines the mass term and provides a method for adding terms.
    :arg function_space: :class:`.FunctionSpace` object. The function
    space that the equation is defined on.

    """
    def __init__(self, function_space):
        self.terms = OrderedDict()
        self.test = TestFunction(function_space)
        self.trial = TrialFunction(function_space)
        self.bcs = None

    def mass_term(self, q):
        return inner(self.test, q)*dx

    def add_term(self, term):
        key = term.__class__.__name__
        self.terms[key] = term


class AdvectionEquation(Equation):
    """
    Class defining the advection equation.

    Creates test and trial functions on the given function space,
    defines the mass term and adds the advection term.
    :arg function_space: :class:`.FunctionSpace` object. The function
    space that the equation is defined on.

    """
    def __init__(self, function_space, state, **kwargs):

        super().__init__(function_space)

        self.add_term(AdvectionTerm(state, self.test, **kwargs))


class ShallowWaterMomentumEquation(Equation):
    """
    Class defining the shallow water momentum equation.

    Creates test and trial functions on the given function space,
    defines the mass term and adds the shallow water momentum terms.
    :arg function_space: :class:`.FunctionSpace` object. The function
    space that the equation is defined on.
    :arg state: :class:`.State` object.

    """
    def __init__(self, function_space, state):

        super().__init__(function_space)

        self.add_term(ShallowWaterPressureGradientTerm(state, self.test))
        self.add_term(ShallowWaterCoriolisTerm(state, self.test))
        self.add_term(VectorInvariantTerm(state, self.test))


class ShallowWaterDepthEquation(AdvectionEquation):
    """
    Class defining the shallow water depth equation.

    Creates test and trial functions on the given function space,
    defines the mass term and adds continuity term.
    :arg function_space: :class:`.FunctionSpace` object. The function
    space that the equation is defined on.
    :arg state: :class:`.State` object.

    """
    def __init__(self, function_space, state):

        super().__init__(function_space, state, equation_form="continuity")


class Equations(object, metaclass=ABCMeta):
    """
    Base equations class.

    Builds compatible finite element function spaces as specified by
    the child class.
    Provides a list of prognostic fields, the mixed function space the
    equations are defined on and when called with the name of a field,
    will return the equation satisfied by that field.
    :arg state: :class:`.State` object.
    :arg family: string specifying the finite element family of the HDiv space
    :arg degree: degree of the DG space

    """

    def __init__(self, state, family, degree):

        self.state = state
        self._build_function_spaces(state.spaces, state.mesh, family, degree)
        state.fields = FieldCreator(self)

        if hasattr(state, "diagnostics"):
            state.diagnostics.register(*self.fieldlist)
        else:
            state.diagnostics = Diagnostics(*self.fieldlist)

    @abstractproperty
    def fieldlist(self):
        pass

    @abstractproperty
    def equations(self):
        pass

    @abstractproperty
    def mixed_function_space(self):
        pass

    @abstractmethod
    def _build_function_spaces(self):
        pass

    def __call__(self, field):
        return self.equations[field]


class ShallowWaterEquations(Equations):
    """
    Class defining the shallow water equations.

    Builds compatible finite element function spaces for the shallow
    water equations.
    Provides a list of prognostic fields, the mixed function space the
    equations are defined on and when called with the name of a field,
    will return the equation satisfied by that field.
    :arg state: :class:`.State` object.
    :arg family: string specifying the finite element family of the HDiv space
    :arg degree: degree of the DG space

    """

    fieldlist = ['u', 'D']

    def __init__(self, state, family, degree):
        super().__init__(state, family, degree)
        self.ueqn = ShallowWaterMomentumEquation(self.u_space, self.state)
        self.Deqn = ShallowWaterDepthEquation(self.D_space, self.state)

    @property
    def mixed_function_space(self):
        return MixedFunctionSpace((self.u_space, self.D_space))

    @property
    def equations(self):
        return {"u": self.ueqn, "D": self.Deqn}

    def _build_function_spaces(self, spaces, mesh, family, degree):
        cell = mesh.ufl_cell().cellname()
        V1_elt = FiniteElement(family, cell, degree+1)
        self.u_space = spaces("HDiv", mesh, V1_elt)
        self.D_space = spaces("DG", mesh, "DG", degree)
