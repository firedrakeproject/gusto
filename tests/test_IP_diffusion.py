from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, Expression, \
    VectorFunctionSpace, File, as_tensor
import pytest


def setup_IPdiffusion(vector, DG):

    dt = 0.01
    L = 10.
    m = PeriodicIntervalMesh(50, L)
    mesh = ExtrudedMesh(m, layers=50, layer_height=0.2)

    fieldlist = ['u','D']
    timestepping = TimesteppingParameters(dt=dt)
    parameters = CompressibleParameters()

    # vertical coordinate and normal
    W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
    W_CG1 = FunctionSpace(mesh, "CG", 1)
    z = Function(W_CG1).interpolate(Expression("x[1]"))
    k = Function(W_VectorCG1).interpolate(Expression(("0.","1.")))

    state = CompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                              family="CG",
                              z=z, k=k,
                              timestepping=timestepping,
                              parameters=parameters,
                              fieldlist=fieldlist)

    if vector:
        if DG:
            Space = VectorFunctionSpace(mesh, "DG", 1)
        else:
            Space = state.V[0]
        f = Function(Space, name="f")
        fexpr = Expression(("exp(-pow(L/2.-x[1],2) - pow(L/2.-x[0],2))", "0.0"), L=L)
    else:
        if DG:
            Space = state.V[1]
        else:
            Space = state.V[2]
        f = Function(Space, name='f')
        fexpr = Expression("exp(-pow(L/2.-x[1],2) - pow(L/2.-x[0],2))", L=L)

    try:
        f.interpolate(fexpr)
    except NotImplementedError:
        f.project(fexpr)

    return state, f


def run(dirname, vector, DG):

    state, f = setup_IPdiffusion(vector, DG)

    if DG:
        direction = [1,2]
    else:
        direction = [1]

    kappa = 0.05
    if vector:
        kappa = as_tensor([[kappa, 0.],[0., kappa]])
    mu = 5.
    dt = state.timestepping.dt
    tmax = 2.5
    t = 0.
    f_diffusion = InteriorPenulty(state, f.function_space(), direction=direction, params={"kappa":kappa, "mu":Constant(mu)})
    outfile = File(path.join(dirname, "IPdiffusion/field_output.pvd"))

    dumpcount = itertools.count()

    outfile.write(f)

    fp1 = Function(f.function_space())

    while t < tmax + 0.5*dt:
        t += dt
        f_diffusion.apply(f, fp1)
        f.assign(fp1)

        if(next(dumpcount) % 25) == 0:
            outfile.write(f)
    return f


@pytest.mark.parametrize("vector", [True, False])
@pytest.mark.parametrize("DG", [True, False])
def test_ipdiffusion(tmpdir, vector, DG):

    dirname = str(tmpdir)
    f = run(dirname, vector, DG)
    assert f.dat.data.max() < 0.7
