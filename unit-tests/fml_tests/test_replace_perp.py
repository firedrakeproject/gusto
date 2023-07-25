from gusto import *
from firedrake import (UnitSquareMesh,
                       SpatialCoordinate,
                       as_vector, TestFunctions,
                       TrialFunctions, solve,
                       inner, dx, errornorm)


def test_replace_perp():

    # The test checks that if the perp operator is applied to the
    # subject of a labelled form, the perp of the subject is found and
    # replaced by the replace_subject function. On the plane this
    # relies on a fix in the replace_subject function that we hope can
    # be removed in future...

    #  set up mesh and function spaces - the subject is defined on a
    #  mixed function space because the bug didn't occur otherwise
    Nx = 5
    mesh = UnitSquareMesh(Nx, Nx)
    domain = Domain(mesh, 0.1, "BDM", 1)
    spaces = [space for space in domain.compatible_spaces]
    W = MixedFunctionSpace(spaces)

    #  set up labelled form with subject u
    w, p = TestFunctions(W)
    U0 = Function(W)
    u0, _ = split(U0)
    form = subject(inner(domain.perp(u0), w)*dx, U0)

    # make a function to replace the subject with and give it some values
    U1 = Function(W)
    u1, _ = U1.split()
    x, y = SpatialCoordinate(mesh)
    u1.interpolate(as_vector([1, 2]))

    u, D = TrialFunctions(W)
    a = inner(u, w)*dx + D*p*dx
    L = form.label_map(all_terms, replace_subject(U1, 0))
    U2 = Function(W)
    solve(a == L.form, U2)

    u2, _ = U2.split()
    U3 = Function(W)
    u3, _ = U3.split()
    u3.interpolate(as_vector([-2, 1]))

    assert errornorm(u2, u3) < 1e-14
