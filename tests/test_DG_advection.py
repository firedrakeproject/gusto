from dcore import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector, assemble, dx
from math import pi

def setup():

    refinements = 3  # number of horizontal cells = 20*(4^refinements)
    R = 1.
    dt = pi/3*0.001

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    global_normal = Expression(("x[0]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[1]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])"))
    mesh.init_cell_orientations(global_normal)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dumplist=(False,True), dumpfreq=150)
    parameters = ShallowWaterParameters()

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=2,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters)

    # interpolate initial conditions
    CG = FunctionSpace(mesh, "CG", 1)
    u0, Ddg, Dcg = Function(state.V[0]), Function(state.V[1]), Function(CG)
    x = SpatialCoordinate(mesh)
    uexpr = as_vector([-x[1], x[0], 0.0])
    Dexpr = Expression("exp(-pow(x[2],2) - pow(x[1],2))")

    u0.project(uexpr)
    Ddg.interpolate(Dexpr)
    Dcg.interpolate(Dexpr)

    state.initialise([u0, Ddg, Dcg])

    # names of fields to dump
    state.fieldlist = ['u', 'D']

    return state, Ddg.function_space(), Dcg

def run():

    state, Vdg, Dcg = setup()
    state.fieldlist.append('dcg')
    state.field_dict['dcg']=Dcg
    
    dt = state.timestepping.dt
    tmax = pi/2.
    t = 0.
    Ddg_advection = DGAdvection(state, Vdg)
    Dcg_advection = EmbeddedDGAdvection(state, Dcg.function_space(), Vdg)
    Dcgp1 = Function(Dcg.function_space())

    state.xn.assign(state.x_init)
    xn_fields = [state.xn.split()[1], Dcg]
    xnp1_fields = [state.xnp1.split()[1], Dcgp1]
    Ddg_advection.ubar.assign(state.xn.split()[0])
    Dcg_advection.ubar.assign(state.xn.split()[0])
    state.dump()

    while t < tmax - 0.5*dt:
        t += dt
        Ddg_advection.apply(xn_fields[0], xnp1_fields[0])
        Dcg_advection.apply(xn_fields[1], xnp1_fields[1])
        state.xn.assign(state.xnp1)
        xn_fields[1].assign(xnp1_fields[1])
        state.dump()

    return state.xn.split()[1]

def test_dgadvection():

    D = run()
    Dend = Function(D.function_space())
    x = SpatialCoordinate(D.function_space().mesh())
    Dexpr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")
    Dend.interpolate(Dexpr)
    Derr = Function(D.function_space()).assign(Dend - D)
    assert(Derr.dat.data.max() < 1.e-2)
    assert(abs(Derr.dat.data.max()) < 1.5e-2)
