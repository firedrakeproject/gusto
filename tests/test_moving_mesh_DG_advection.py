from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    as_vector
import pytest
from math import pi


def setup_DGadvection(dirname, continuity=False):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)
    R = 1.
    dt = pi/3*0.01

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    fieldlist = ['u','D']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/DGadvection")

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = Function(state.V[0])
    x = SpatialCoordinate(mesh)
    uexpr = as_vector([-x[1], x[0], 0.0])
    u0.project(uexpr)

    f = Function(state.V[1])
    fexpr = Expression("exp(-pow(x[2],2) - pow(x[1],2))")
    f_end = Function(state.V[1])
    f_end_expr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")

    f.interpolate(fexpr)
    f_end.interpolate(f_end_expr)
    state.initialise([u0, f])

    advection_list = []
    f_advection = DGAdvection(state, f.function_space(), continuity=continuity, scale=0.5)
    advection_list.append((f_advection,1))
    vexpr = Expression(("0.0", "0.2*cos(2*pi*x[0])*sin(pi*t/(20.*dt))", "0.0"), R=R, dt=dt, t=0.0)
    v = Function(state.V[0]).project(vexpr)

    stepper = MovingMeshAdvectionTimestepper(state, advection_list, v, vexpr)
    return stepper, f_end


def run(dirname, continuity=False):

    stepper, f_end = setup_DGadvection(dirname, continuity)

    tmax = pi/2.

    x_end = stepper.run(t=0, tmax=tmax, x_end=True)
    f = x_end.split()[1]

    f_err = Function(f.function_space()).assign(f_end - f)
    return f_err


@pytest.mark.parametrize("continuity", [False])
def test_dgadvection(tmpdir, continuity):

    dirname = str(tmpdir)
    f_err = run(dirname, continuity)
    assert(abs(f_err.dat.data.max()) < 2.5e-2)
