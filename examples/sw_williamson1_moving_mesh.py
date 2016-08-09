from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector, File, cos
from math import pi
# setup resolution and timestepping parameters for convergence test
# ref_dt = {3:3000., 4:1500., 5:750., 6:375}
ref_dt = {4:1800.}

# setup shallow water parameters
R = 6371220.
day = 24.*60.*60.
u_0 = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters()
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.iteritems():

    dirname = "sw_mm_W1_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname, dumpfreq=10, dumplist_latlon=['D','u'])

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters,
                              diagnostics=diagnostics,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0, D0 = Function(state.V[0]), Function(state.V[1])
    x = SpatialCoordinate(mesh)
    u_max = Constant(u_0)
    R0 = Constant(R)
    uexpr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
    u0.project(uexpr)
    Dexpr = Expression("fabs(x[2]) < R ? R*acos(-sqrt(1-x[2]*x[2]/R/R)*x[1]/sqrt(x[0]*x[0]+x[1]*x[1])) > rc ? 0.0 : 0.5*h0*(1.+cos(pi*(R/rc)*acos(-sqrt(1-x[2]*x[2]/R/R)*x[1]/sqrt(x[0]*x[0]+x[1]*x[1])))) : 0.5*pi*R > rc ? 0.0 : 0.5*h0*(1.+cos(pi*0.5*pi*R/rc))", R=R, rc=R/3., h0=1000.)
    D0.interpolate(Dexpr)
    state.initialise([u0, D0])

    # Coriolis expression
    Omega = Constant(parameters.Omega)
    fexpr = 2*Omega*x[2]/R0
    V = FunctionSpace(mesh, "CG", 1)
    state.f = Function(V).interpolate(fexpr)  # Coriolis frequency (1/s)

    advection_dict = {}
    advection_dict["D"] = DGAdvection(state, state.V[1], continuity=True)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state)

    # build time stepper
    vexpr = as_vector([0.0, x[2]/R, -x[1]/R])
    Vu = VectorFunctionSpace(mesh, "DG", 2)
    uadv = Function(Vu).interpolate(u0)
    moving_mesh_advection = MovingMeshAdvection(state, advection_dict, vexpr, uadv=uadv)
    stepper = MovingMeshAdvectionTimestepper(state, advection_dict, moving_mesh_advection)

    stepper.run(t=0, tmax=12*day)
