from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, Constant, \
    as_vector
from math import pi
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 3000.}
    tmax = 5*day

# setup shallow water parameters
R = 6371220.
H = 5960.
u_0 = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.items():

    dirname = "mm_ot_sw_W2_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=2)
    mesh.init_cell_orientations(SpatialCoordinate(mesh))

    timestepping = TimesteppingParameters(dt=dt, move_mesh=True)
    output = OutputParameters(dirname=dirname, dumplist_latlon=['D', 'D_error'], steady_state_error_fields=['D', 'u'])

    state = State(mesh, horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")

    x0, y0, z0 = SpatialCoordinate(mesh)
    R0 = Constant(R)
    x = R0*x0/sqrt(x0*x0 + y0*y0 + z0*z0)  # because coords can behave unexpectedly
    y = R0*y0/sqrt(x0*x0 + y0*y0 + z0*z0)  # away from nodes, e.g. at quad points
    z = R0*z0/sqrt(x0*x0 + y0*y0 + z0*z0)

    u_max = Constant(u_0)
    uexpr = as_vector([-u_max*y/R0, u_max*x/R0, 0.0])
    h0 = Constant(H)
    Omega = Constant(parameters.Omega)
    g = Constant(parameters.g)
    Dexpr = h0 - ((R0 * Omega * u_max + u_max*u_max/2.0)*(z*z/(R0*R0)))/g
    # Coriolis expression
    fexpr = 2*Omega*z/R0
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", V)
    f.interpolate(fexpr)  # Coriolis frequency (1/s)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0), ('D', D0)])

    ueqn = EulerPoincare(state, u0.function_space())
    Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")

    state.uexpr = uexpr

    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state)

    def initialise_fn():
        state.fields("u").project(uexpr)
        state.fields("D").interpolate(Dexpr)

    monitor = MonitorFunction(state.fields("D"))
    mesh_generator = OptimalTransportMeshGenerator(mesh, monitor)

    mesh_generator.get_first_mesh(initialise_fn)

    # build time stepper
    stepper = Timestepper(state, advected_fields, linear_solver,
                          sw_forcing, mesh_generator=mesh_generator)

    stepper.run(t=0, tmax=tmax)
