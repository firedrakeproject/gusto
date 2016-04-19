from dcore import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    as_vector, VectorFunctionSpace, File
import itertools
import pytest
from math import pi


def setup_DGadvection(vector=False):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)
    R = 1.
    dt = pi/3*0.001

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    fieldlist = ['u','D']
    timestepping = TimesteppingParameters(dt=dt)

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=2,
                              family="BDM",
                              timestepping=timestepping,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = Function(state.V[0], name="velocity")
    x = SpatialCoordinate(mesh)
    uexpr = as_vector([-x[1], x[0], 0.0])
    u0.project(uexpr)

    if vector:
        DGSpace = VectorFunctionSpace(mesh, "DG", 1)
        Space = VectorFunctionSpace(mesh, "CG", 1)
        f = Function(Space, name="f")
        fexpr = Expression(("exp(-pow(x[2],2) - pow(x[1],2))", "0.0", "0.0"))
        f_end = Function(Space)
        f_end_expr = Expression(("exp(-pow(x[2],2) - pow(x[0],2))","0","0"))
    else:
        DGSpace = FunctionSpace(mesh, "DG", 1)
        Space = FunctionSpace(mesh, "CG", 1)
        f = Function(Space, name='f')
        fexpr = Expression("exp(-pow(x[2],2) - pow(x[1],2))")
        f_end = Function(Space)
        f_end_expr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")

    f.interpolate(fexpr)
    f_end.interpolate(f_end_expr)

    return state, DGSpace, u0, f, f_end


def run(dirname, continuity=False, vector=False):

    state, dgfunctionspace, u0, f, f_end = setup_DGadvection(vector)

    dt = state.timestepping.dt
    tmax = pi/4.
    t = 0.
    f_advection = EmbeddedDGAdvection(state, f.function_space(), dgfunctionspace, continuity=continuity)

    fp1 = Function(f.function_space())
    f_advection.ubar.assign(u0)

    dumpcount = itertools.count()
    outfile = File(path.join(dirname, "field_output.pvd"))
    outfile.write(f)

    while t < tmax + 0.5*dt:
        t += dt
        for i in range(2):
            f_advection.apply(f, fp1)
            f.assign(fp1)

        if(next(dumpcount) % 150) == 0:
            outfile.write(f)

    f_err = Function(f.function_space()).assign(f_end - f)
    return f_err


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("continuity", [False, True])
def test_embedded_dg(tmpdir, vector, continuity):

    dirname = str(tmpdir)
    f_err = run(dirname, vector, continuity)
    assert(abs(f_err.dat.data.max()) < 1.5e-2)
