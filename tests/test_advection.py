from gusto import *
from firedrake import IcosahedralSphereMesh, PeriodicIntervalMesh, \
    ExtrudedMesh, Expression, SpatialCoordinate, \
    as_vector, VectorFunctionSpace, File, sin
import itertools
import pytest
from math import pi

error = {"slice": 7e-2, "sphere": 2.5e-2}


def setup_advection(dirname, geometry, time_discretisation, ibp_twice, continuity, vector, spatial_opts=None):

    output = OutputParameters(dirname=dirname, dumplist=["f"], dumpfreq=15)
    if geometry is "sphere":
        refinements = 3  # number of horizontal cells = 20*(4^refinements)
        R = 1.
        dt = pi/3*0.01
        tmax = pi/2.

        mesh = IcosahedralSphereMesh(radius=R,
                                     refinement_level=refinements, degree=3)
        global_normal = Expression(("x[0]", "x[1]", "x[2]"))
        mesh.init_cell_orientations(global_normal)

        fieldlist = ['u','D']
        timestepping = TimesteppingParameters(dt=dt)

        state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                                  family="BDM",
                                  timestepping=timestepping,
                                  output=output,
                                  fieldlist=fieldlist)
        x = SpatialCoordinate(mesh)
        uexpr = as_vector([-x[1], x[0], 0.0])
        if vector:
            VectorDGSpace = VectorFunctionSpace(mesh, "DG", 1)
            f = Function(VectorDGSpace, name="f")
            fexpr = Expression(("exp(-pow(x[2],2) - pow(x[1],2))", "0.0", "0.0"))
            f_end = Function(VectorDGSpace)
            f_end_expr = Expression(("exp(-pow(x[2],2) - pow(x[0],2))","0","0"))
        else:
            f = Function(state.V[1], name='f')
            fexpr = Expression("exp(-pow(x[2],2) - pow(x[1],2))")
            f_end = Function(state.V[1])
            f_end_expr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")

    elif geometry is "slice":
        nlayers = 25  # horizontal layers
        columns = 25  # number of columns
        L = 1.0
        m = PeriodicIntervalMesh(columns, L)

        H = 1.0  # Height position of the model top
        mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
        dt = 0.01
        tmax = 2.5

        # Spaces for initialising k and z
        W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
        W_CG1 = FunctionSpace(mesh, "CG", 1)

        # vertical coordinate and normal
        z = Function(W_CG1).interpolate(Expression("x[1]"))
        k = Function(W_VectorCG1).interpolate(Expression(("0.","1.")))

        fieldlist = ['u','rho', 'theta']
        timestepping = TimesteppingParameters(dt=dt)
        parameters = CompressibleParameters()

        state = CompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                                  family="CG",
                                  z=z, k=k,
                                  timestepping=timestepping,
                                  output=output,
                                  parameters=parameters,
                                  fieldlist=fieldlist)

        uexpr = as_vector([1.0, 0.0])

        if spatial_opts is not None and "supg" in spatial_opts:
            if len(kwargs.get("direction")) > 0:
                space = state.V[2]
            else:
                space = W_CG1
        else:
            space = state.V[1]
        f = Function(space, name='f')
        x = SpatialCoordinate(mesh)
        fexpr = sin(2*pi*x[0])*sin(2*pi*x[1])
        f_end = Function(space)
        f_end_expr = sin(2*pi*(x[0]-0.5))*sin(2*pi*x[1])

    # interpolate initial conditions
    u0 = Function(state.V[0], name="velocity")
    u0.project(uexpr)
    D0 = Function(state.V[-1], name="unused").interpolate(Constant(0))
    state.initialise([u0, D0])
    f.interpolate(fexpr)
    state.field_dict["f"] = f
    f_end.interpolate(f_end_expr)

    if spatial_opts is not None:
        fequation = AdvectionEquation(state, f.function_space(), continuity=continuity, **spatial_opts)
    else:
        fequation = AdvectionEquation(state, f.function_space(), continuity=continuity)
    if time_discretisation is "ssprk":
        f_advection = SSPRK3(state, f, fequation)
    elif time_discretisation is "implicit_midpoint":
        f_advection = ImplicitMidpoint(state, f, fequation)

    advection_dict = {}
    advection_dict["f"] = f_advection
    timestepper = AdvectionTimestepper(state, advection_dict)

    return timestepper, tmax, f_end


def run(dirname, geometry, time_discretisation, ibp_twice, continuity, vector, spatial_opts=None):

    timestepper, tmax, f_end = setup_advection(dirname, geometry, time_discretisation, ibp_twice, continuity, vector, spatial_opts=None)

    f_dict = timestepper.run(0, tmax, x_end=["f"])
    f = f_dict["f"]

    f_err = Function(f.function_space()).assign(f_end - f)
    return f_err


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("time_discretisation", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("ibp_twice", [False, True])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("continuity", [False, True])
def test_advection_dg(tmpdir, geometry, time_discretisation, ibp_twice, vector, continuity):

    dirname = str(tmpdir)
    f_err = run(dirname, geometry, time_discretisation, ibp_twice, vector, continuity)
    assert(abs(f_err.dat.data.max()) < error[geometry])


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("time_discretisation", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("ibp_twice", [False, True])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("continuity", [False, True])
def test_advection_embedded_dg(tmpdir, geometry, time_discretisation, ibp_twice, vector, continuity):

    dirname = str(tmpdir)
    f_err = run(dirname, geometry, time_discretisation, ibp_twice, vector, continuity)
    assert(abs(f_err.dat.data.max()) < error[geometry])


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("time_discretisation", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("ibp_twice", [False, True])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("continuity", [False, True])
@pytest.mark.parametrize("direction", [[], [1]])
def test_advection_supg(tmpdir, geometry, time_discretisation, ibp_twice, vector, continuity, direction):

    dirname = str(tmpdir)
    f_err = run(dirname, geometry, time_discretisation, ibp_twice, vector, continuity, spatial_opts={"supg":{"direction":direction}})
    assert(abs(f_err.dat.data.max()) < error[geometry])
