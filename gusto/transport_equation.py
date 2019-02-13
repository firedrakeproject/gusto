from enum import Enum
from firedrake import (Function, TestFunction, TestFunctions, FacetNormal,
                       dx, dot, grad, div, jump, avg, dS, dS_v, dS_h, inner,
                       ds, ds_v, ds_t, ds_b,
                       outer, sign, cross, CellNormal,
                       curl, Constant)
from gusto.form_manipulation_labelling import advection, advecting_velocity, subject
=======
                       ds_v, ds_t, ds_b, VectorElement, as_ufl,
                       outer, sign, cross, CellNormal, Constant,
                       curl, BrokenElement, FunctionSpace)
from gusto.configuration import logger, DEBUG, SUPGOptions
>>>>>>> origin


__all__ = ["IntegrateByParts", "advection_form", "continuity_form"]


class IntegrateByParts(Enum):
    NEVER = 0
    ONCE = 1
    TWICE = 2


def is_cg(V):
    """
    Function to check is a given space, V, is CG. Broken elements are
    always discontinuous; for vector elements we check the names of
    the sobolev spaces of the subelements and for all other elements
    we just check the sobolev space name.
    """
    ele = V.ufl_element()
    if isinstance(ele, BrokenElement):
        return False
    elif type(ele) == VectorElement:
        return all([e.sobolev_space().name == "H1" for e in ele._sub_elements])
    else:
        return V.ufl_element().sobolev_space().name == "H1"


def is_dg(V):
    """
    Function to check is a given space, V, is DG. Broken elements are
    always discontinuous; for vector elements we check the names of
    the sobolev spaces of the subelements and for all other elements
    we just check the sobolev space name.
    """
    ele = V.ufl_element()
    if isinstance(ele, BrokenElement):
        return True
    elif type(ele) == VectorElement:
        return all([e.sobolev_space().name == "L2" for e in ele._sub_elements])
    else:
        return V.ufl_element().sobolev_space().name == "L2"


def surface_measures(V):
    """
    Function returning the correct surface measures to use for the
    given function space, V, based on its continuity and also on
    whether the underlying mesh is extruded.
    """
    # CG spaces don't require surface terms
    if is_cg(V):
        return None
    else:
        # for extruded meshes we have to unpick things
        if V.extruded:
            dim = V.mesh().topological_dimension()
            ele = V.ufl_element()

            # Broken elements are discontinuous so need dS_v and dS_h
            if is_dg(V):
                return (dS_v + dS_h)

            # Figure out which spaces we have
            elif type(ele) == VectorElement:
                spaces = [ele._sub_elements[i].sobolev_space().name
                          for i in range(dim)]
            else:
                spaces = [ele.sobolev_space()[i].name for i in range(dim)]

            # Based on the space names, these ones are discontinuous
            # enough to require surface terms
            discont = [i for i, name in enumerate(spaces) if name in ["L2", "HDiv"]]
            # as the mesh is extruded, the dim-1 element of discont
            # tells us about the vertical discontinuity
            is_discontinuous_v = dim-1 in discont
            # check now for horizontal discontinuity
            is_discontinuous_h = any([i in discont for i in range(dim-1)])

            # return surface measures
            if is_discontinuous_h and is_discontinuous_v:
                return (dS_v + dS_h)
            elif is_discontinuous_h:
                return dS_v
            elif is_discontinuous_v:
                return dS_h
            else:
                raise ValueError("Sorry, we have failed to figure out the surface measures for this space")
        else:
            # if we're here we're discontinuous in some way, but not
            # on an extruded mesh so things are easy
            return dS


def setup_functions(state, V, idx):

    X = Function(V)
    if len(V) > 1:
        if idx is None:
            raise ValueError("If V is a mixed function space you must specify the index of the space this form is defined on")
        test = TestFunctions(V)[idx]
        q = X.split()[idx]
        ubar = Function(V.sub(0))
    else:
        test = TestFunction(V)
        q = X
        ubar = Function(state.spaces("HDiv"))
    return X, test, q, ubar


def advection_form(state, V, idx=None, *,
                   ibp=IntegrateByParts.ONCE, outflow=None):
    """
    The equation is assumed to be in the form:

    q_t + L(q) = 0

    where q is the (scalar or vector) field to be solved for.

    :arg state: :class:`.State` object.
    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    """

    X, test, q, ubar = setup_functions(state, V, idx)
    dS, ds = surface_measures(q.function_space())

    if ibp == IntegrateByParts.ONCE:
        L = -inner(div(outer(test, ubar)), q)*dx
    else:
        L = inner(outer(test, ubar), grad(q))*dx

    if dS is not None and ibp != IntegrateByParts.NEVER:
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
        self.dS = surface_measures(V)

        L += dot(jump(test), (un('+')*q('+') - un('-')*q('-')))*dS

        if ibp == IntegrateByParts.TWICE:
            L -= (inner(test('+'), dot(ubar('+'), n('+'))*q('+'))
                  + inner(test('-'), dot(ubar('-'), n('-'))*q('-')))*dS

    if outflow is not None:
        L += test*un*q*ds

    form = subject(advection(advecting_velocity(L, ubar)), X)
    return form


def linear_continuity_form(state, V, idx=None, *, qbar=None):

    X, test, _, ubar = setup_functions(state, V, idx)

    form = subject(advection(advecting_velocity(Constant(qbar)*test*div(ubar)*dx, ubar)), X)
    return form


def continuity_form(state, V, idx=None, *,
                    ibp=IntegrateByParts.ONCE):

    X, test, q, ubar = setup_functions(state, V, idx)
    dS, ds = surface_measures(q.function_space())

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

    form = subject(advection(advecting_velocity(L, ubar)), X)
    return form


def advection_vector_manifold_form(state, V, idx=None, *,
                                   ibp=IntegrateByParts.ONCE, outflow=None):

    X, test, q, ubar = setup_functions(state, V, idx)
    dS, ds = surface_measures(q.function_space())

    n = FacetNormal(state.mesh)
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

    L = un('+')*inner(test('-'), n('+')+n('-'))*inner(q('+'), n('+'))*dS
    L += un('-')*inner(test('+'), n('+')+n('-'))*inner(q('-'), n('-'))*dS

    form = advection_form(state, V, idx, ibp=ibp) + subject(advection(advecting_velocity(L, ubar)), X)
    return form


def vector_invariant_form(state, V, idx=None, *,
                          ibp=IntegrateByParts.ONCE):
    """
    Defines the vector invariant form of the vector advection term.

    :arg state: :class:`.State` object.
    :arg V: Function space
    :arg ibp: (optional) string, stands for 'integrate by parts' and can
              take the value None, "once" or "twice". Defaults to "once".
    """

    X, test, q, ubar = setup_functions(state, V, idx)
    dS, ds = surface_measures(q.function_space())

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
        gradperp = lambda u: perp(grad(u))

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
                + jump(inner(test, perp(ubar))*perp(q), n)*dS
            )

    L -= 0.5*div(test)*inner(q, ubar)*dx
    form = subject(advection(advecting_velocity(L, ubar)), X)
    return form


def kinetic_energy_form(state, V, idx=None):

    X, test, q, ubar = setup_functions(state, V, idx)

    form = subject(advection(advecting_velocity(0.5*div(test)*inner(q, ubar)*dx, ubar)), X)
    return form


def advection_equation_circulation_form(state, V, idx=None, *,
                                        ibp=IntegrateByParts.ONCE):
    """
    Defining the circulation form of the vector advection term.

    :arg state: :class:`.State` object.
    :arg V: Function space
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    """
    form = (
        vector_invariant_form(state, V, idx, ibp=ibp)
        - kinetic_energy_form(state, V, idx)
    )
    return form
