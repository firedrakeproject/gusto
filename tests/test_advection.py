from gusto import *
from firedrake import IcosahedralSphereMesh, PeriodicIntervalMesh, \
    ExtrudedMesh, Expression, SpatialCoordinate, \
    as_vector, VectorFunctionSpace, sin
import pytest
from math import pi

error = {"slice": 7e-2, "sphere": 2.5e-2}


def setup_advection(dirname, geometry, time_discretisation, ibp, equation_form, vector, spatial_opts=None):

    output = OutputParameters(dirname=dirname, dumplist=["f"], dumpfreq=15)
    if geometry == "sphere":
        refinements = 3  # number of horizontal cells = 20*(4^refinements)
        R = 1.
        dt = pi/3*0.01
        tmax = pi/2.

        mesh = IcosahedralSphereMesh(radius=R,
                                     refinement_level=refinements, degree=3)
        global_normal = Expression(("x[0]", "x[1]", "x[2]"))
        mesh.init_cell_orientations(global_normal)

        fieldlist = ['u', 'D']
        timestepping = TimesteppingParameters(dt=dt)

        state = State(mesh, horizontal_degree=1,
                      family="BDM",
                      timestepping=timestepping,
                      output=output,
                      fieldlist=fieldlist)
        x = SpatialCoordinate(mesh)
        uexpr = as_vector([-x[1], x[0], 0.0])
        u0 = state.fields("u")
        u0.project(uexpr)

        if vector:
            space = VectorFunctionSpace(mesh, "DG", 1)
            fexpr = Expression(("exp(-pow(x[2],2) - pow(x[1],2))", "0.0", "0.0"))
            f_end_expr = Expression(("exp(-pow(x[2],2) - pow(x[0],2))", "0.0", "0.0"))
        else:
            space = state.spaces("DG")
            fexpr = Expression("exp(-pow(x[2],2) - pow(x[1],2))")
            f_end_expr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")
        f = state.fields("f", space)
        f_end = Function(space)

    elif geometry == "slice":
        nlayers = 25  # horizontal layers
        columns = 25  # number of columns
        L = 1.0
        m = PeriodicIntervalMesh(columns, L)

        H = 1.0  # Height position of the model top
        mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
        dt = 0.01
        tmax = 2.5

        fieldlist = ['u', 'rho', 'theta']
        timestepping = TimesteppingParameters(dt=dt)
        parameters = CompressibleParameters()

        state = State(mesh, vertical_degree=1, horizontal_degree=1,
                      family="CG",
                      timestepping=timestepping,
                      output=output,
                      parameters=parameters,
                      fieldlist=fieldlist)

        uexpr = as_vector([1.0, 0.0])
        u0 = state.fields("u")
        u0.project(uexpr)

        if spatial_opts is not None:
            if "supg_params" in spatial_opts:
                # if the direction list is empty we are testing SUPG for a
                # continuous space, else we are testing the hybrid SUPG /
                # DG upwind scheme for the theta space
                if spatial_opts["supg_params"]["dg_direction"] is None:
                    space = FunctionSpace(mesh, "CG", 1)
                else:
                    space = state.spaces("HDiv_v")
            elif "embedded_dg" in spatial_opts:
                space = state.spaces("HDiv_v")
        else:
            space = state.spaces("DG")
        f = state.fields("f", space)
        x = SpatialCoordinate(mesh)
        fexpr = sin(2*pi*x[0])*sin(2*pi*x[1])
        f_end = Function(space)
        f_end_expr = sin(2*pi*(x[0]-0.5))*sin(2*pi*x[1])

    # interpolate initial conditions
    f.interpolate(fexpr)
    f_end.interpolate(f_end_expr)
    state.initialise({'u': u0, 'f': f})

    if spatial_opts is None:
        fequation = AdvectionEquation(state, f.function_space(), ibp=ibp, equation_form=equation_form)
    elif "supg_params" in spatial_opts:
        fequation = SUPGAdvection(state, f.function_space(), ibp=ibp, equation_form=equation_form, supg_params=spatial_opts["supg_params"])
    elif "embedded_dg" in spatial_opts:
        if spatial_opts["embedded_dg"]["space"] == "Broken":
            fequation = EmbeddedDGAdvection(state, f.function_space(), ibp=ibp, equation_form=equation_form)
        elif spatial_opts["embedded_dg"]["space"] == "DG":
            fequation = EmbeddedDGAdvection(state, f.function_space(), ibp=ibp, equation_form=equation_form, Vdg=space)

    if time_discretisation == "ssprk":
        f_advection = SSPRK3(state, f, fequation)
    elif time_discretisation == "implicit_midpoint":
        f_advection = ThetaMethod(state, f, fequation)

    advection_dict = {}
    advection_dict["f"] = f_advection
    timestepper = AdvectionTimestepper(state, advection_dict)

    return timestepper, tmax, f_end


def run(dirname, geometry, time_discretisation, ibp, equation_form, vector, spatial_opts=None):

    timestepper, tmax, f_end = setup_advection(dirname, geometry, time_discretisation, ibp, equation_form, vector, spatial_opts=spatial_opts)

    f_dict = timestepper.run(0, tmax, x_end=["f"])
    f = f_dict["f"]

    f_err = Function(f.function_space()).assign(f_end - f)
    return f_err


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("time_discretisation", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("ibp", ["once", "twice"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("vector", [False, True])
def test_advection_dg(tmpdir, geometry, time_discretisation, ibp, equation_form, vector):

    dirname = str(tmpdir)
    f_err = run(dirname, geometry, time_discretisation, ibp, equation_form, vector)
    assert(abs(f_err.dat.data.max()) < error[geometry])


@pytest.mark.parametrize("ibp", ["once", "twice"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("space", ["Broken", "DG"])
def test_advection_embedded_dg(tmpdir, ibp, equation_form, space):

    geometry = "slice"
    time_discretisation = "ssprk"
    vector = False
    dirname = str(tmpdir)
    f_err = run(dirname, geometry, time_discretisation, ibp, equation_form, vector, spatial_opts={"embedded_dg": {"space": space}})
    assert(abs(f_err.dat.data.max()) < error[geometry])


@pytest.mark.parametrize("time_discretisation", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("ibp", [None, "twice"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("vector", [False, True])
def test_advection_supg(tmpdir, time_discretisation, ibp, equation_form, vector):
    geometry = "slice"
    if ibp is None:
        direction = None
    else:
        direction = "horizontal"
    dirname = str(tmpdir)
    f_err = run(dirname, geometry, time_discretisation, ibp, equation_form, vector, spatial_opts={"supg_params": {"dg_direction": direction}})
    assert(abs(f_err.dat.data.max()) < error[geometry])
