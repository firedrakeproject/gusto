from math import pi
from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector, parameters

parameters["pyop2_options"]["lazy_evaluation"] = False
# setup resolution and timestepping parameters for convergence test
# ref_dt = {3:3000., 4:1500., 5:750., 6:375}
ref_dt = {3:1000.}

# setup shallow water parameters
R = 6371220.
day = 24.*60.*60.
u_0 = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber()]

for ref_level, dt in ref_dt.iteritems():

    dirname = "mm_ot_sw_W1cont_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    timestepping = TimesteppingParameters(dt=dt, move_mesh=True)
    output = OutputParameters(dirname=dirname, dumpfreq=12, dumplist_latlon=['D','u'])

    state = State(mesh, horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = state.fields.u
    D0 = state.fields.D
    x = SpatialCoordinate(mesh)
    u_max = Constant(u_0)
    R0 = Constant(R)
    uexpr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
    u0.project(uexpr)
    Dexpr = Expression("R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0)) < rc ? (h0/2.0)*(1 + cos(pi*R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0))/rc)) : 0.0", R=R, rc=R/3., h0=1000., x0=0.0, x1=-R, x2=0.0)
    D0.interpolate(Dexpr)
    state.initialise({'u': u0, 'D': D0})

    # Coriolis expression
    Omega = Constant(parameters.Omega)
    fexpr = 2*Omega*x[2]/R0
    V = FunctionSpace(mesh, "CG", 1)
    state.f = Function(V).interpolate(fexpr)  # Coriolis frequency (1/s)

    eqn_form = "continuity"
    Deqn = AdvectionEquation(state, D0.function_space(), equation_form=eqn_form)

    vscale = 10.0
    dt = state.timestepping.dt
    state.uexpr = uexpr

    advection_dict = {}
    advection_dict["D"] = SSPRK3(state, D0, Deqn)
    advection_dict["u"] = NoAdvection(state, u0)

    def initialise_fn():
        state.fields("u").project(uexpr)
        state.fields("D").interpolate(Dexpr)

    monitor = MonitorFunction(state.fields("D"))
    mesh_generator = OptimalTransportMeshGenerator(mesh, monitor)

    mesh_generator.get_first_mesh(initialise_fn)

    # build time stepper
    stepper = AdvectionTimestepper(state, advection_dict, mesh_generator=mesh_generator)

    stepper.run(t=0, tmax=12*day)
