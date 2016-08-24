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
                                 refinement_level=refinements,
                                 degree=3)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    fieldlist = ['u','D']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/DGadvection", dumpfreq=10)

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = Function(state.V[0])
    x = SpatialCoordinate(mesh)
    uexpr = as_vector([-x[1], x[0], 0.0])

    f = Function(state.V[1])
    fexpr = Expression("exp(-pow(x[2],2) - pow(x[1],2))")
    f_end_expr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")

    f.interpolate(fexpr)
    state.initialise([u0, f])

    advection_dict = {}
    advection_dict["D"] = DGAdvection(state, f.function_space(), continuity=continuity)
    vscale = Constant(0.5)
    vexpr = vscale*as_vector([0.0, x[2], -x[1]])
    V = VectorFunctionSpace(mesh, "DG", 2)
    uadv = Function(V)
    moving_mesh_advection = MovingMeshAdvection(state, advection_dict, vexpr, uadv, uexpr)

    stepper = MovingMeshAdvectionTimestepper(state, advection_dict, moving_mesh_advection)
    return stepper, f_end_expr


def run(dirname, continuity=False):

    stepper, f_end_expr = setup_DGadvection(dirname, continuity)

    tmax = pi/2.

    f_dict = stepper.run(t=0, tmax=tmax, x_end=["D"])
    f = f_dict["D"]
    f_end = Function(f.function_space()).interpolate(f_end_expr)
    f_err = Function(f.function_space()).assign(f_end - f)
    return f_err


@pytest.mark.parametrize("continuity", [True, False])
def test_dgadvection(tmpdir, continuity):

    dirname = str(tmpdir)
    f_err = run(dirname, continuity)
    assert(abs(f_err.dat.data.max()) < 2.5e-2)
