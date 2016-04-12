from dcore import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    as_vector, VectorFunctionSpace, File
import itertools
from math import pi


def setup_DGadvection(dirname):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)
    R = 1.
    dt = pi/3*0.001

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    global_normal = Expression(("x[0]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[1]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])"))
    mesh.init_cell_orientations(global_normal)

    fieldlist = ['u','D']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dumpfreq=150, dirname='results/tests/'+dirname)
    parameters = ShallowWaterParameters()

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=2,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0, Ddg = Function(state.V[0], name="velocity"), Function(state.V[1], name="D")
    VectorDGSpace = VectorFunctionSpace(mesh, "DG", 1)
    vdg = Function(VectorDGSpace, name="v")
    x = SpatialCoordinate(mesh)
    uexpr = as_vector([-x[1], x[0], 0.0])
    Dexpr = Expression("exp(-pow(x[2],2) - pow(x[1],2))")
    vexpr = Expression(("exp(-pow(x[2],2) - pow(x[1],2))", "0.0", "0.0"))

    u0.project(uexpr)
    vdg.interpolate(vexpr)
    Ddg.interpolate(Dexpr)

    return state, u0, Ddg, vdg


def run(continuity):

    state, u0, Ddg, vdg = setup_DGadvection(dirname="DGAdvection/continuity" + str(continuity))

    dt = state.timestepping.dt
    tmax = pi/4.
    t = 0.
    Ddg_advection = DGAdvection(state, Ddg.function_space(), continuity=continuity)
    vdg_advection = DGAdvection(state, vdg.function_space(), continuity=continuity)

    Ddgp1 = Function(Ddg.function_space())
    vdgp1 = Function(vdg.function_space())
    Ddg_advection.ubar.assign(u0)
    vdg_advection.ubar.assign(u0)

    dumpcount = itertools.count()
    outfile = File(state.output.dirname+".pvd")
    outfile.write(Ddg, vdg)

    while t < tmax + 0.5*dt:
        t += dt
        for i in range(2):
            Ddg_advection.apply(Ddg, Ddgp1)
            Ddg.assign(Ddgp1)
            vdg_advection.apply(vdg, vdgp1)
            vdg.assign(vdgp1)

        if(next(dumpcount) % state.output.dumpfreq) == 0:
            outfile.write(Ddg, vdg)

    return Ddg, vdg


def test_dgadvection():

    D, v = run(continuity=False)
    Dend = Function(D.function_space())
    Dexpr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")
    Dend.interpolate(Dexpr)
    Derr = Function(D.function_space()).assign(Dend - D)
    Vend = Function(v.function_space())
    Vexpr = Expression(("exp(-pow(x[2],2) - pow(x[0],2))","0","0"))
    Vend.interpolate(Vexpr)
    Verr = Function(v.function_space()).assign(Vend - v)

    errfile = File("errF.pvd")
    errfile.write(Derr,Verr)
    print abs(Derr.dat.data.max())
    print abs(Verr.dat.data.max())
    assert(abs(Derr.dat.data.max()) < 1.5e-2)
    assert(abs(Verr.dat.data.max()) < 1.5e-2)


def test_dgadvection_continuity():

    D, v = run(continuity=True)
    Dend = Function(D.function_space())
    Dexpr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")
    Dend.interpolate(Dexpr)
    Derr = Function(D.function_space()).assign(Dend - D)
    Vend = Function(v.function_space())
    Vexpr = Expression(("exp(-pow(x[2],2) - pow(x[0],2))","0","0"))
    Vend.interpolate(Vexpr)
    Verr = Function(v.function_space()).assign(Vend - v)

    errfile = File("errT.pvd")
    errfile.write(Derr,Verr)
    print abs(Derr.dat.data.max())
    print abs(Verr.dat.data.max())
    assert(abs(Derr.dat.data.max()) < 1.5e-2)
    assert(abs(Verr.dat.data.max()) < 1.5e-2)
