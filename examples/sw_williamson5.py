from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 900., 4: 450., 5: 225., 6: 112.5}
    tmax = 50*day

# setup shallow water parameters
R = 6371220.
H = 5960.
u_0 = 20.  # Maximum amplitude of the zonal wind (m/s)

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.iteritems():

    dirname = "sw_W5_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname, dumplist_latlon=['D'])
    diagnostic_fields = [Sum('D', 'topography')]

    state = State(mesh, horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = state.fields('u')
    D0 = state.fields('D')
    x = SpatialCoordinate(mesh)
    u_max = Constant(u_0)
    R0 = Constant(R)
    uexpr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
    h0 = Constant(H)
    Omega = Constant(parameters.Omega)
    g = Constant(parameters.g)
    Dexpr = Expression("h0 - ((R0 * Omega * u0 + pow(u0,2)/2.0)*(x[2]*x[2]/(R0*R0)))/g - (2000 * (1 - sqrt(fmin(pow(pi/9.0,2),pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2)+pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0)))", h0=5960, R0=R0, Omega=Omega, u0=20.0, g=g)
    bexpr = Expression("2000 * (1 - sqrt(fmin(pow(pi/9.0,2),pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2)+pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0))", R0=R0)

    # Coriolis expression
    fexpr = 2*Omega*x[2]/R0
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", V)
    f.interpolate(fexpr)  # Coriolis frequency (1/s)
    b = state.fields("topography", D0.function_space())
    b.interpolate(bexpr)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise({'u': u0, 'D': D0})

    ueqn = EulerPoincare(state, u0.function_space())
    Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
    advection_dict = {}
    advection_dict["u"] = ThetaMethod(state, u0, ueqn)
    advection_dict["D"] = SSPRK3(state, D0, Deqn)

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state)

    # build time stepper
    stepper = Timestepper(state, advection_dict, linear_solver,
                          sw_forcing)

    stepper.run(t=0, tmax=tmax)
