from gusto import *
from firedrake import IcosahedralSphereMesh, PeriodicIntervalMesh, \
    ExtrudedMesh, SpatialCoordinate, \
    as_vector, VectorFunctionSpace, sin, exp, Function, FunctionSpace
import pytest
from math import pi


@pytest.fixture
def state(tmpdir, geometry):
    output = OutputParameters(dirname=str(tmpdir), dumplist=["f"], dumpfreq=15)

    if geometry == "sphere":
        mesh = IcosahedralSphereMesh(radius=1,
                                     refinement_level=3,
                                     degree=1)
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
        family = "BDM"
        vertical_degree = None
        fieldlist = ["u", "D"]
        dt = pi/3. * 0.01
        uexpr = as_vector([-x[1], x[0], 0.0])

    if geometry == "slice":
        m = PeriodicIntervalMesh(25, 1.)
        mesh = ExtrudedMesh(m, layers=25, layer_height=1./25.)
        family = "CG"
        vertical_degree = 1
        fieldlist = ["u", "rho", "theta"]
        dt = 0.01
        x = SpatialCoordinate(mesh)
        uexpr = as_vector([1.0, 0.0])

    timestepping = TimesteppingParameters(dt=dt)
    state = State(mesh,
                  vertical_degree=vertical_degree,
                  horizontal_degree=1,
                  family=family,
                  timestepping=timestepping,
                  output=output,
                  fieldlist=fieldlist)

    u0 = state.fields("u")
    u0.project(uexpr)
    return state


@pytest.fixture
def f_init(geometry, state):
    x = SpatialCoordinate(state.mesh)
    if geometry == "sphere":
        fexpr = exp(-x[2]**2 - x[1]**2)
    if geometry == "slice":
        fexpr = sin(2*pi*x[0])*sin(2*pi*x[1])
    return fexpr


@pytest.fixture
def f_end(geometry, state):
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
    return {"slice": 7e-2,
            "sphere": 2.5e-2}[geometry]


def run(state, time_discretisation, fequation, tmax):

    f = state.fields("f")
    if time_discretisation == "ssprk":
        f_advection = SSPRK3(state, f, fequation)
    elif time_discretisation == "implicit_midpoint":
        f_advection = ThetaMethod(state, f, fequation)

    advected_fields = [("f", f_advection)]
    timestepper = AdvectionTimestepper(state, advected_fields)

    timestepper.run(0, tmax)

    return timestepper.state.fields("f")


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("time_discretisation", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("ibp", ["once", "twice"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("vector", [False, True])
def test_advection_dg(geometry, time_discretisation, ibp,
                      equation_form, vector, error, state,
                      f_init, tmax, f_end):
    if "vector":
        if geometry == "slice":
            pytest.skip("broken")
        f_space = VectorFunctionSpace(state.mesh, "DG", 1)
        fexpr = as_vector([f_init, 0., 0.])
        f_end_expr = as_vector([f_end, 0., 0.])
    else:
        f_space = state.spaces("DG")
        fexpr = f_init
        f_end_expr = f_end
    f = state.fields("f", f_space)
    f.interpolate(fexpr)
    f_err = Function(f.function_space()).interpolate(f_end_expr)

    fequation = AdvectionEquation(state, f.function_space(),
                                  ibp=ibp, equation_form=equation_form)
    f_end = run(state, time_discretisation, fequation, tmax)
    f_err -= f_end
    assert(abs(f_err.dat.data.max()) < error)


@pytest.mark.parametrize("geometry", ["slice"])
@pytest.mark.parametrize("time_discretisation", ["ssprk"])
@pytest.mark.parametrize("ibp", ["once", "twice"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("space", ["Broken", "DG"])
def test_advection_embedded_dg(geometry, time_discretisation, ibp,
                               equation_form, space, error,
                               state, f_init, tmax, f_end):

    f_space = state.spaces("HDiv_v")
    f = state.fields("f", f_space)
    f.interpolate(f_init)
    f_err = Function(f.function_space()).interpolate(f_end)

    if space == "Broken":
        fequation = EmbeddedDGAdvection(state, f.function_space(), ibp=ibp,
                                        equation_form=equation_form)
    elif space == "DG":
        fequation = EmbeddedDGAdvection(state, f.function_space(), ibp=ibp,
                                        equation_form=equation_form,
                                        Vdg=state.spaces("DG"))
    f = run(state, time_discretisation, fequation, tmax)
    f_err -= f
    assert(abs(f_err.dat.data.max()) < error)


@pytest.mark.parametrize("geometry", ["slice"])
@pytest.mark.parametrize("time_discretisation", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("ibp", [None, "twice"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("vector", [False, True])
def test_advection_supg(geometry, time_discretisation, ibp, equation_form, vector, error, state):

    if ibp is None:
        space = FunctionSpace(mesh, "CG", 1)
        fequation = SUPGAdvection(state, f.function_space(), ibp=ibp,
                                  equation_form=equation_form)
    else:
        space = state.spaces("HDiv_v")
        fequation = SUPGAdvection(state, f.function_space(), ibp=ibp,
                                  equation_form=equation_form,
                                  supg_params={"dg_direction": "horizontal"})

    f_err = run(dirname)
    assert(abs(f_err.dat.data.max()) < error)
