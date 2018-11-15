from firedrake import (Function, TestFunction, inner, dx, div,
                       SpatialCoordinate, sqrt, FunctionSpace,
                       MixedFunctionSpace)
from gusto.form_manipulation_labelling import subject, prognostic_variable
from gusto.transport_equation import (vector_invariant_equation,
                                      continuity_equation)
from gusto.state import build_spaces


def shallow_water_equations(state, family, degree):

    g = state.parameters.g
    Omega = state.parameters.Omega
    x = SpatialCoordinate(state.mesh)
    R = sqrt(inner(x, x))
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(state.mesh, "CG", 1)
    f = Function(V).interpolate(fexpr)
    state.fields("coriolis", f)

    Vu, VD = build_spaces(state, family, degree)
    state.spaces.W = MixedFunctionSpace((Vu, VD))
    u = state.fields("u", Vu)
    D = state.fields("D", VD)
    state.diagnostics.register("u")
    state.diagnostics.register("D")

    u_adv = prognostic_variable(vector_invariant_equation(state, Vu), "u")

    w = TestFunction(Vu)
    coriolis_term = prognostic_variable(subject(-f*inner(w, state.perp(u))*dx, u), "u")
    pressure_gradient_term = prognostic_variable(subject(g*div(w)*D*dx, D), "u")

    u_eqn = u_adv + coriolis_term + pressure_gradient_term

    D_eqn = prognostic_variable(continuity_equation(state, VD), "D")

    return u_eqn + D_eqn
