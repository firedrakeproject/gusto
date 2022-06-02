from firedrake import (Function, FacetNormal,
                       dx, dot, grad, div, jump, avg, dS, dS_v, dS_h, inner,
                       ds_v, ds_t, ds_b,
                       outer, sign, cross, CellNormal,
                       curl)
from gusto.configuration import IntegrateByParts, TransportEquationType
from gusto.labels import advection, advecting_velocity, ibp_label


__all__ = ["advection_form", "continuity_form", "vector_invariant_form", "vector_manifold_advection_form", "kinetic_energy_form", "advection_equation_circulation_form", "linear_continuity_form"]


def linear_advection_form(state, test, qbar):

    ubar = Function(state.spaces("HDiv"))

    L = test*dot(ubar, state.k)*dot(state.k, grad(qbar))*dx

    form = advecting_velocity(L, ubar)

    return advection(form, TransportEquationType.advective)


def linear_continuity_form(state, test, qbar):

    Vu = state.spaces("HDiv")
    ubar = Function(Vu)

    L = qbar*test*div(ubar)*dx

    form = advecting_velocity(L, ubar)

    return advection(form, TransportEquationType.conservative)


def advection_form(state, test, q, ibp=IntegrateByParts.ONCE, outflow=False):

    if outflow and ibp == IntegrateByParts.NEVER:
        raise ValueError("outflow is True and ibp is None are incompatible options")
    Vu = state.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)

    if ibp == IntegrateByParts.ONCE:
        L = -inner(div(outer(test, ubar)), q)*dx
    else:
        L = inner(outer(test, ubar), grad(q))*dx

    if ibp != IntegrateByParts.NEVER:
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

        L += dot(jump(test), (un('+')*q('+') - un('-')*q('-')))*dS_

        if ibp == IntegrateByParts.TWICE:
            L -= (inner(test('+'), dot(ubar('+'), n('+'))*q('+'))
                  + inner(test('-'), dot(ubar('-'), n('-'))*q('-')))*dS_

    if outflow:
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
        L += test*un*q*(ds_v + ds_t + ds_b)

    form = advecting_velocity(L, ubar)

    return ibp_label(advection(form, TransportEquationType.advective), ibp)


def continuity_form(state, test, q, ibp=IntegrateByParts.ONCE, outflow=False):

    if outflow and ibp == IntegrateByParts.NEVER:
        raise ValueError("outflow is True and ibp is None are incompatible options")
    Vu = state.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)

    if ibp == IntegrateByParts.ONCE:
        L = -inner(grad(test), outer(q, ubar))*dx
    else:
        L = inner(test, div(outer(q, ubar)))*dx

    if ibp != IntegrateByParts.NEVER:
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

        L += dot(jump(test), (un('+')*q('+') - un('-')*q('-')))*dS_

        if ibp == IntegrateByParts.TWICE:
            L -= (inner(test('+'), dot(ubar('+'), n('+'))*q('+'))
                  + inner(test('-'), dot(ubar('-'), n('-'))*q('-')))*dS_

    if outflow:
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
        L += test*un*q*(ds_v + ds_t + ds_b)

    form = advecting_velocity(L, ubar)

    return ibp_label(advection(form, TransportEquationType.conservative), ibp)


def vector_manifold_advection_form(state, test, q, ibp=IntegrateByParts.ONCE, outflow=False):

    L = advection_form(state, test, q, ibp, outflow)

    Vu = state.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)
    n = FacetNormal(state.mesh)
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    L += un('+')*inner(test('-'), n('+')+n('-'))*inner(q('+'), n('+'))*dS_
    L += un('-')*inner(test('+'), n('+')+n('-'))*inner(q('-'), n('-'))*dS_

    return L


def vector_manifold_continuity_form(state, test, q, ibp=IntegrateByParts.ONCE, outflow=False):

    L = continuity_form(state, test, q, ibp, outflow)

    Vu = state.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)
    n = FacetNormal(state.mesh)
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    L += un('+')*inner(test('-'), n('+')+n('-'))*inner(q('+'), n('+'))*dS_
    L += un('-')*inner(test('+'), n('+')+n('-'))*inner(q('-'), n('-'))*dS_

    form = advecting_velocity(L, ubar)

    return advection(form)


def vector_invariant_form(state, test, q, ibp=IntegrateByParts.ONCE):

    Vu = state.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)
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
                    both(cross(n, cross(ubar, test))))*dS_
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
                        perp_u_upwind(q))*dS_
            )
        else:
            L = (
                (-inner(test, div(perp(q))*perp(ubar)))*dx
                - inner(jump(inner(test, perp(ubar)), n),
                        perp_u_upwind(q))*dS_
                + jump(inner(test,
                             perp(ubar))*perp(q), n)*dS_
            )

    L -= 0.5*div(test)*inner(q, ubar)*dx

    form = advecting_velocity(L, ubar)

    return advection(form, TransportEquationType.vector_invariant)


def kinetic_energy_form(state, test, q):

    ubar = Function(state.spaces("HDiv"))
    L = 0.5*div(test)*inner(q, ubar)*dx

    form = advecting_velocity(L, ubar)

    return advection(form, TransportEquationType.vector_invariant)


def advection_equation_circulation_form(state, test, q,
                                        ibp=IntegrateByParts.ONCE):

    form = (
        vector_invariant_form(state, test, q, ibp=ibp)
        - kinetic_energy_form(state, test, q)
    )

    return form
