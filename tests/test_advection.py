from gusto import *
from firedrake import (IcosahedralSphereMesh, PeriodicIntervalMesh,
                       ExtrudedMesh, SpatialCoordinate, as_vector,
                       VectorFunctionSpace, sin, exp, Function, FunctionSpace)
import pytest
from math import pi


@pytest.fixture
def state(tmpdir, geometry):
    """
    returns an instance of the State class, having set up either spherical
    geometry or 2D vertical slice geometry
    """

    output = OutputParameters(dirname=str(tmpdir), dumplist=["f"], dumpfreq=15)

    if geometry == "sphere":
        mesh = IcosahedralSphereMesh(radius=1,
                                     refinement_level=3,
                                     degree=1)
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
        family = "BDM"
        vertical_degree = None
        dt = pi/3. * 0.01
        uexpr = as_vector([-x[1], x[0], 0.0])

    if geometry == "slice":
        m = PeriodicIntervalMesh(15, 1.)
        mesh = ExtrudedMesh(m, layers=15, layer_height=1./15.)
        family = "CG"
        vertical_degree = 1
        dt = 0.01
        x = SpatialCoordinate(mesh)
        uexpr = as_vector([1.0, 0.0])

    state = State(mesh, dt,
                  output=output)

    build_spaces(state, family, 1, vertical_degree)
    u = state.fields("u", space=state.spaces("HDiv"))
    u.project(uexpr)

    return state


@pytest.fixture
def f_init(geometry, state):
    """
    returns an expression for the initial condition
    """
    x = SpatialCoordinate(state.mesh)
    if geometry == "sphere":
        fexpr = exp(-x[2]**2 - x[1]**2)
    if geometry == "slice":
        fexpr = sin(2*pi*x[0])*sin(2*pi*x[1])
    return fexpr


@pytest.fixture
def f_end(geometry, state):
    """
    returns an expression for the expected final state
    """
    x = SpatialCoordinate(state.mesh)
    if geometry == "sphere":
        fexpr = exp(-x[2]**2 - x[0]**2)
    if geometry == "slice":
        fexpr = sin(2*pi*(x[0]-0.5))*sin(2*pi*x[1])
    return fexpr


@pytest.fixture
def tmax(geometry):
    return {"slice": 2.5,
            "sphere": pi/2}[geometry]


@pytest.fixture
def error(geometry):
    """
    returns the max expected error (based on past runs)
    """
    return {"slice": 7e-2,
            "sphere": 2.5e-2}[geometry]


def run(state, equations, schemes, tmax):

    timestepper = Timestepper(state, equations=equations, schemes=schemes)
    timestepper.run(0, tmax)
    return timestepper.state.fields


def check_errors(ans, error, end_fields, field_names):
    for fname in field_names:
        f = end_fields(fname)
        f -= ans
        assert(abs(f.dat.data.max()) < error)


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
def test_advection_dg(geometry, error, state,
                      f_init, tmax, f_end):
    """
    This tests the DG advection discretisation for both scalar and vector
    fields in 2D slice and spherical geometry.
    """
    # set up function spaces
    fspace = state.spaces("DG")
    vspace = VectorFunctionSpace(state.mesh, "DG", 1)

    # expression for vector initial and final conditions
    vec_expr = [0.]*state.mesh.geometric_dimension()
    vec_expr[0] = f_init
    vec_expr = as_vector(vec_expr)
    vec_end_expr = [0.]*state.mesh.geometric_dimension()
    vec_end_expr[0] = f_end
    vec_end_expr = as_vector(vec_end_expr)

    # functions containing expected values at tmax
    f_end = Function(fspace).interpolate(f_end)
    vec_end = Function(vspace).interpolate(vec_end_expr)

    s = "_"
    uadv = state.fields('u')
    eqns = []
    schemes = []

    # setup scalar fields
    scalar_fields = []
    for ibp in [IntegrateByParts.ONCE, IntegrateByParts.TWICE]:
        for time_discretisation in ["ssprk", "im"]:
            # create functions and initialise them
            fname = s.join(("f", ibp.name, time_discretisation))
            eqns.append(
                (fname, AdvectionEquation(state, fname, fspace, uadv, ibp=ibp))
            )
            f = state.fields(fname, space=fspace)
            f.interpolate(f_init)
            scalar_fields.append(fname)
            if time_discretisation == "ssprk":
                schemes.append((fname, SSPRK3(state)))
            elif time_discretisation == "im":
                schemes.append((fname, ThetaMethod(state)))

    # setup vector fields
    vector_fields = []
    for ibp in [IntegrateByParts.ONCE, IntegrateByParts.TWICE]:
        for time_discretisation in ["im"]:
            # create functions and initialise them
            fname = s.join(("vecf", ibp.name, time_discretisation))
            eqns.append(
                (fname, AdvectionEquation(state, fname, vspace, uadv, ibp=ibp))
            )
            f = state.fields(fname, space=vspace)
            f.interpolate(vec_expr)
            vector_fields.append(fname)
            if time_discretisation == "ssprk":
                schemes.append((fname, SSPRK3(state)))
            elif time_discretisation == "im":
                schemes.append((fname, ThetaMethod(state)))

    end_fields = run(state, eqns, schemes, tmax)

    check_errors(f_end, error, end_fields, scalar_fields)
    check_errors(vec_end, error, end_fields, vector_fields)


@pytest.mark.parametrize("geometry", ["slice"])
def test_advection_embedded_dg(geometry, error, state, f_init, tmax, f_end):
    """
    This tests the embedded DG advection scheme for scalar fields
    in slice geometry.
    """
    fspace = state.spaces("HDiv_v")
    f_end = Function(fspace).interpolate(f_end)

    s = "_"
    uadv = state.fields('u')
    eqns = []
    schemes = []
    opts = {"broken": EmbeddedDGOptions(),
            "dg": EmbeddedDGOptions(embedding_space=state.spaces("DG"))}

    # setup scalar fields
    scalar_fields = []
    for ibp in [IntegrateByParts.ONCE, IntegrateByParts.TWICE]:
        for equation_form in ["advective"]:
            for space in ["broken", "dg"]:
                # create functions and initialise them
                fname = s.join(("f", ibp.name, equation_form, space))
                eqns.append(
                    (fname,
                     AdvectionEquation(state, fname, fspace, uadv, ibp=ibp))
                )
                f = state.fields(fname, space=fspace)
                f.interpolate(f_init)
                scalar_fields.append(fname)
                schemes.append((fname, SSPRK3(state, options=opts[space])))

    end_fields = run(state, eqns, schemes, tmax)
    check_errors(f_end, error, end_fields, scalar_fields)


@pytest.mark.parametrize("geometry", ["slice"])
def test_advection_supg(geometry, error, state, f_init, tmax, f_end):
    """
    This tests the embedded DG advection scheme for scalar and vector fields
    in slice geometry.
    """
    s = "_"
    uadv = state.fields('u')
    eqns = []
    schemes = []

    cgspace = FunctionSpace(state.mesh, "CG", 1)
    fspace = state.spaces("HDiv_v")
    vcgspace = VectorFunctionSpace(state.mesh, "CG", 1)
    vspace = state.spaces("HDiv")

    # expression for vector initial and final conditions
    vec_expr = [0.]*state.mesh.geometric_dimension()
    vec_expr[0] = f_init
    vec_expr = as_vector(vec_expr)
    vec_end_expr = [0.]*state.mesh.geometric_dimension()
    vec_end_expr[0] = f_end
    vec_end_expr = as_vector(vec_end_expr)

    cg_end = Function(cgspace).interpolate(f_end)
    hdiv_v_end = Function(fspace).interpolate(f_end)
    vcg_end = Function(vcgspace).interpolate(vec_end_expr)
    hdiv_end = Function(vspace).project(vec_end_expr)

    supg_opts = SUPGOptions()
    # setup cg scalar fields
    cg_scalar_fields = []
    ibp = IntegrateByParts.NEVER
    for equation_form in ["advective"]:
        for time_discretisation in ["ssprk"]:
            # create functions and initialise them
            fname = s.join(("f", equation_form, time_discretisation))
            eqns.append(
                (fname, AdvectionEquation(state, fname, cgspace, uadv, ibp=ibp))
            )
            f = state.fields(fname, space=cgspace)
            f.interpolate(f_init)
            cg_scalar_fields.append(fname)
            if time_discretisation == "ssprk":
                schemes.append((fname, SSPRK3(state, options=supg_opts)))
            elif time_discretisation == "im":
                schemes.append((fname, ThetaMethod(state, options=supg_opts)))

    # setup cg vector fields
    cg_vector_fields = []
    ibp = IntegrateByParts.NEVER
    for equation_form in ["advective"]:
        for time_discretisation in ["ssprk"]:
            # create functions and initialise them
            fname = s.join(("fvec", equation_form, time_discretisation))
            eqns.append(
                (fname,
                 AdvectionEquation(state, fname, vcgspace, uadv, ibp=ibp))
            )
            f = state.fields(fname, space=vcgspace)
            f.interpolate(vec_expr)
            cg_vector_fields.append(fname)
            if time_discretisation == "ssprk":
                schemes.append((fname, SSPRK3(state, options=supg_opts)))
            elif time_discretisation == "im":
                schemes.append((fname, ThetaMethod(state, options=supg_opts)))

    # setup HDiv_v fields
    hdiv_v_fields = []
    ibp = IntegrateByParts.TWICE
    for equation_form in ["advective"]:
        for time_discretisation in ["ssprk"]:
            # create functions and initialise them
            fname = s.join(("f", ibp.name, equation_form, time_discretisation))
            eqns.append(
                (fname, AdvectionEquation(state, fname, fspace, uadv, ibp=ibp))
            )
            f = state.fields(fname, space=fspace)
            f.interpolate(f_init)
            hdiv_v_fields.append(fname)
            if time_discretisation == "ssprk":
                schemes.append((fname, SSPRK3(state, options=supg_opts)))
            elif time_discretisation == "im":
                schemes.append((fname, ThetaMethod(state, options=supg_opts)))

    # setup HDiv fields
    hdiv_fields = []
    ibp = IntegrateByParts.TWICE
    for equation_form in ["advective"]:
        for time_discretisation in ["ssprk"]:
            # create functions and initialise them
            fname = s.join(("fvec", ibp.name, equation_form, time_discretisation))
            eqns.append(
                (fname, AdvectionEquation(state, fname, vspace, uadv, ibp=ibp))
            )
            f = state.fields(fname, space=vspace)
            f.project(vec_expr)
            hdiv_fields.append(fname)
            if time_discretisation == "ssprk":
                schemes.append((fname, SSPRK3(state, options=supg_opts)))
            elif time_discretisation == "im":
                schemes.append((fname, ThetaMethod(state, options=supg_opts)))

    end_fields = run(state, eqns, schemes, tmax)
    check_errors(cg_end, error, end_fields, cg_scalar_fields)
    check_errors(hdiv_v_end, error, end_fields, hdiv_v_fields)
    check_errors(vcg_end, error, end_fields, cg_vector_fields)
    check_errors(hdiv_end, error, end_fields, hdiv_fields)
