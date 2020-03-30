from enum import Enum
from firedrake import (Function, TestFunction, FacetNormal,
                       dx, dot, grad, div, jump, avg, dS, dS_v, dS_h, inner,
                       ds_v, ds_t, ds_b, VectorElement,
                       outer, sign, cross, CellNormal,
                       curl, BrokenElement)
from gusto.form_manipulation_labelling import (advection, advecting_velocity,
                                               subject)


__all__ = ["IntegrateByParts", "advection_form", "continuity_form"]


class IntegrateByParts(Enum):
    NEVER = 0
    ONCE = 1
    TWICE = 2


def surface_measures(V):
    """
    Function returning the correct surface measures to use for the
    given function space, V, based on its continuity and also on
    whether the underlying mesh is extruded.
    """
    if V.extruded:
        # if the mesh is extruded we need both the vertical and
        # horizontal interior facets
        return (dS_v + dS_h)
    else:
        # if we're here we're discontinuous in some way, but not
        # on an extruded mesh so things are easy
        return dS


def setup_functions(state, V):
    q = Function(V)
    test = TestFunction(V)
    ubar = Function(state.spaces("HDiv"))
    dS_ = surface_measures(V)

    return q, test, ubar, dS_


def linear_advection_form(state, V, qbar, ibp=IntegrateByParts.NEVER):

    _, test, ubar, _ = setup_functions(state, V)

    L = test*dot(ubar, state.k)*dot(state.k, grad(qbar))*dx

    form = advecting_velocity(L, ubar)

    return advection(form)


def linear_continuity_form(state, V, qbar):

    _, test, ubar, _ = setup_functions(state, V)
    n = FacetNormal(state.mesh)

    L = (-dot(grad(test), ubar)*qbar*dx + jump(ubar*test, n)*avg(qbar)*dS)

    form = advecting_velocity(L, ubar)

    return advection(form)


def advection_form(state, V, *, ibp=IntegrateByParts.ONCE, outflow=False):

    if outflow and ibp == IntegrateByParts.NEVER:
        raise ValueError("outflow is True and ibp is None are incompatible options")

    q, test, ubar, dS = setup_functions(state, V)

    if ibp == IntegrateByParts.ONCE:
        L = -inner(div(outer(test, ubar)), q)*dx
    else:
        L = inner(outer(test, ubar), grad(q))*dx

    if dS is not None and ibp != IntegrateByParts.NEVER:
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

        L += dot(jump(test), (un('+')*q('+') - un('-')*q('-')))*dS

        if ibp == IntegrateByParts.TWICE:
            L -= (inner(test('+'), dot(ubar('+'), n('+'))*q('+'))
                  + inner(test('-'), dot(ubar('-'), n('-'))*q('-')))*dS

    if outflow:
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
        L += test*un*q*(ds_v + ds_t + ds_b)

    form = advecting_velocity(L, ubar)
    form = subject(form, q)

    return advection(form)


def continuity_form(state, V, *, ibp=IntegrateByParts.ONCE, outflow=False):

    if outflow and ibp == IntegrateByParts.NEVER:
        raise ValueError("outflow is True and ibp is None are incompatible options")

    q, test, ubar, dS = setup_functions(state, V)

    if ibp == IntegrateByParts.ONCE:
        L = -inner(grad(test), outer(q, ubar))*dx
    else:
        L = inner(test, div(outer(q, ubar)))*dx

    if dS is not None and ibp != IntegrateByParts.NEVER:
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

        L += dot(jump(test), (un('+')*q('+') - un('-')*q('-')))*dS

        if ibp == IntegrateByParts.TWICE:
            L -= (inner(test('+'), dot(ubar('+'), n('+'))*q('+'))
                  + inner(test('-'), dot(ubar('-'), n('-'))*q('-')))*dS

    if outflow:
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
        L += test*un*q*(ds_v + ds_t + ds_b)

    form = advecting_velocity(L, ubar)
    form = subject(form, q)

    return advection(form)


def vector_manifold_advection_form(state, V, *, ibp=IntegrateByParts.ONCE, outflow=False):

    q, test, ubar, dS = setup_functions(state, V)

    L = advection_form(state, V, ibp, outflow)

    n = FacetNormal(state.mesh)
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    L += un('+')*inner(test('-'), n('+')+n('-'))*inner(q('+'), n('+'))*dS
    L += un('-')*inner(test('+'), n('+')+n('-'))*inner(q('-'), n('-'))*dS

    return L


def vector_manifold_continuity_form(state, V, *, ibp=IntegrateByParts.ONCE, outflow=False):

    q, test, ubar, dS = setup_functions(state, V)

    L = continuity_form(state, V, ibp, outflow)

    n = FacetNormal(state.mesh)
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    L += un('+')*inner(test('-'), n('+')+n('-'))*inner(q('+'), n('+'))*dS
    L += un('-')*inner(test('+'), n('+')+n('-'))*inner(q('-'), n('-'))*dS

    form = advecting_velocity(L, ubar)
    form = subject(form, q)

    return advection(form)


def vector_invariant_form(state, V, *, ibp=IntegrateByParts.ONCE):

    q, test, ubar, dS = setup_functions(state, V)

    n = FacetNormal(state.mesh)
    Upwind = 0.5*(sign(dot(ubar, n))+1)

    if state.mesh.topological_dimension() == 3:

        if ibp != IntegrateByParts.ONCE:
            raise NotImplementedError

        # <w,curl(u) cross ubar + grad( u.ubar)>
        # =<curl(u),ubar cross w> - <div(w), u.ubar>
        # =<u,curl(ubar cross w)> -
        #      <<u_upwind, [[n cross(ubar cross w)cross]]>>

        both = lambda u: 2*avg(u)

        L = (
            inner(q, curl(cross(ubar, test)))*dx
            - inner(both(Upwind*q),
                    both(cross(n, cross(ubar, test))))*dS
        )

    else:

        perp = state.perp
        if state.on_sphere:
            outward_normals = CellNormal(state.mesh)
            perp_u_upwind = lambda q: Upwind('+')*cross(outward_normals('+'), q('+')) + Upwind('-')*cross(outward_normals('-'), q('-'))
        else:
            perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))

        if ibp == IntegrateByParts.ONCE:
            L = (
                -inner(perp(grad(inner(test, perp(ubar)))), q)*dx
                - inner(jump(inner(test, perp(ubar)), n),
                        perp_u_upwind(q))*dS
            )
        else:
            L = (
                (-inner(test, div(perp(q))*perp(ubar)))*dx
                - inner(jump(inner(test, perp(ubar)), n),
                        perp_u_upwind(q))*dS
                + jump(inner(test,
                             perp(ubar))*perp(q), n)*dS
            )

    L -= 0.5*div(test)*inner(q, ubar)*dx

    form = advecting_velocity(L, ubar)
    form = subject(form, q)

    return advection(form)


def kinetic_energy_form(state, V):

    q, test, ubar, _ = setup_functions(state, V)

    L = 0.5*div(test)*inner(q, ubar)*dx

    form = advecting_velocity(L, ubar)
    form = subject(form, q)

    return advection(form)


def advection_equation_circulation_form(state, V, *,
                                        ibp=IntegrateByParts.ONCE):

    form = (
        vector_invariant_form(state, V, ibp=ibp)
        - kinetic_energy_form(state, V)
    )

    return form
