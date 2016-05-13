from dcore import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, Expression, \
    VectorFunctionSpace, FunctionSpace, as_vector, File, sin, SpatialCoordinate
import itertools
import pytest
from math import pi


def setup_SUPGadvection():

    nlayers = 100  # horizontal layers
    columns = 100  # number of columns
    L = 1.0
    m = PeriodicIntervalMesh(columns, L)

    H = 1.0  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    # Space for initialising velocity
    W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
    W_CG1 = FunctionSpace(mesh, "CG", 1)

    # vertical coordinate and normal
    z = Function(W_CG1).interpolate(Expression("x[1]"))
    k = Function(W_VectorCG1).interpolate(Expression(("0.","1.")))

    fieldlist = ['u','rho', 'theta']
    timestepping = TimesteppingParameters(dt=0.005)
    parameters = CompressibleParameters()

    state = CompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                              family="CG",
                              z=z, k=k,
                              timestepping=timestepping,
                              parameters=parameters,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = Function(state.V[0], name="velocity")
    uexpr = as_vector([1.0, 0.0])
    u0.project(uexpr)

    f = Function(state.V[2], name='f')
    x = SpatialCoordinate(mesh)
    f_expr = sin(2*pi*x[0])*sin(2*pi*x[1])
    f.interpolate(f_expr)
    f_end = Function(state.V[2])
    f_end_expr = sin(2*pi*(x[0]-0.5))*sin(2*pi*x[1])
    f_end.interpolate(f_end_expr)

    return state, u0, f, f_end


def run(dirname, direction):

    state, u0, f, f_end = setup_SUPGadvection()

    dt = state.timestepping.dt
    tmax = 2.5
    t = 0.
    f_advection = SUPGAdvection(state, f.function_space(), direction=direction)

    fp1 = Function(f.function_space())
    f_advection.ubar.assign(u0)

    dumpcount = itertools.count()
    outfile = File(path.join(dirname, "field_output.pvd"))
    outfile.write(f)

    while t < tmax + 0.5*dt:
        t += dt
        f_advection.apply(f, fp1)
        f.assign(fp1)

        if(next(dumpcount) % 80) == 0:
            outfile.write(f)

    f_err = Function(f.function_space()).assign(f_end - f)
    return f_err


@pytest.mark.parametrize("direction", [None, 1])
def test_supgadvection(tmpdir, direction):

    dirname = str(tmpdir)
    f_err = run(dirname, direction)
    assert(abs(f_err.dat.data.max()) < 5.0e-2)
