from dcore import *
from firedrake import PeriodicSquareMesh, Expression, \
    as_vector, VectorFunctionSpace


def setup_moving_mesh(dirname):

    dt = 0.01
    L = 10.
    mesh = PeriodicSquareMesh(50,50,L)

    fieldlist = ['u','D']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/moving_mesh", dumpfreq=20)
    diagnostic_fields = [CourantNumber()]

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              fieldlist=fieldlist,
                              diagnostic_fields=diagnostic_fields)

    # interpolate initial conditions
    Vcg = VectorFunctionSpace(mesh, "CG", 1)
    u0 = Function(Vcg, name="velocity")
    uexpr = as_vector([1.0, 0.0])
    u0.project(uexpr)

    # setup mesh velocity
    vexpr = Expression(("0.0","2*sin(x[1]*pi/L)*sin(2*x[0]*pi/L)*sin(0.5*pi*t)"), t=0, L=L)
    v = Function(mesh.coordinates.function_space()).interpolate(vexpr)

    f = Function(state.V[1], name='f')
    fexpr = Expression("exp(-pow((L/2.-x[1]),2) - pow((L/2.-x[0]),2))", L=L)
    f.interpolate(fexpr)

    return state, u0, f, v, vexpr


def run(dirname):

    state, u0, f, v, vexpr = setup_moving_mesh(dirname)

    dt = state.timestepping.dt
    tmax = 10.
    t = 0.
    f_advection = DGAdvection(state, state.V[1], scale=0.5, continuity=False)
    f_advection.ubar.project(u0 - v)

    fp1 = Function(f.function_space())
    fstar = Function(f.function_space())

    while t < tmax + 0.5*dt:
        t += dt
        # First advection step on current mesh
        f_advection.ubar.project(u0 - v)
        f_advection.apply(f, fstar)

        # Move mesh
        x = state.mesh.coordinates
        vexpr.t = t
        v.interpolate(vexpr)
        x += dt*v
        v.interpolate(vexpr)

        # Second advection step on new mesh
        f_advection.ubar.project(u0 - v)
        f_advection.apply(fstar, fp1)

        f.assign(fp1)

        state.dump()


def test_moving_mesh_courant(tmpdir):
    dirname = str(tmpdir)
    run(dirname)
