from gusto import *
from firedrake import (PeriodicRectangleMesh, conditional, TestFunction,
                       TrialFunction, exp, Constant, sqrt)
# set up mesh
Lx = 10000e3
Ly = 10000e3
delta = 80e3
nx = int(Lx/delta)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction="x")

# set up parameters
day = 24*60*60
dt = 200
tau = 200
H = 30.
g = 10
fexpr = Constant(0)
beta = 0
lamda_r = 1.1e-5
tau_e = 1e6
q_0 = 3
q_g = 3
alpha = 60
gamma = 5 # 0 if humidity is a passive tracer
nu_u = (1.5e4)/4
nu_D = (1e4)/4
nu_Q = (2e4)/4
L_d = sqrt(sqrt(g*H)/1e-20 + beta)

parameters=MoistShallowWaterParameters(H=H, g=g, gamma=gamma, tau_e=tau_e,
                                       tau=tau, q_0=q_0, q_g=q_g, alpha=alpha,
                                       lamda_r=lamda_r)

dirname="tracer_sw_narrowgaussian_test"

output = OutputParameters(dirname=dirname, dumpfreq=100)

diagnostic_fields = [CourantNumber()]

state = State(mesh,
              dt=dt,
              output=output,
              diagnostic_fields = diagnostic_fields,
              parameters=parameters)

x, y = SpatialCoordinate(mesh)

diffusion_options = [
    ("u", DiffusionParameters(kappa=nu_u, mu=10./delta)),
    ("D", DiffusionParameters(kappa=nu_D, mu=10./delta)),
    ("Q_mixing_ratio", DiffusionParameters(kappa=nu_Q, mu=10./delta))]

moisture_variable = WaterVapour(name="Q", space="DG",
                                variable_type=TracerVariableType.mixing_ratio,
                                transport_flag=True,
                                transport_eqn=TransportEquationType.advective)

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr,
                             no_normal_flow_bc_ids=[1,2],
                             active_tracers = [moisture_variable],
                             diffusion_options=diffusion_options)

print([field.name() for field in state.fields])
u0 = state.fields("u")
D0 = state.fields("D")
Q0 = state.fields("Q_mixing_ratio")

# gaussian = 10*exp(-((x-0.5*Lx)**2/2.5e11 + (y-0.5*Ly)**2/2.5e11))
gaussian = 10*exp(-((x-0.5*Lx)**2/(80e3)**2 + (y-0.5*Ly)**2/(80e3)**2))
lump = 10 * exp(-(sqrt(((x-0.5*Lx)+1e6)**2 + ((y-0.5*Ly)+1e6)**2)/0.05*Ly)**2)
Q0.interpolate(q_g * Constant(1 - 1e-4))
D0.interpolate(Constant(H) - 0.01 * gaussian)


stepper = Timestepper(state, ((eqns, RK4(state)),),)

stepper.run(t=0, tmax=4*day)
