# This example is to test the implementation of the Bouchut et al moist
# shallow water framework.
from gusto import *
from firedrake import (PeriodicSquareMesh, conditional, TestFunction,
                       TrialFunction, exp, Constant, sqrt)

# set up mesh
Lx = 10000e3
Ly = 10000e3
delta = 80e3
nx = int(Lx/delta)

mesh = PeriodicSquareMesh(nx, nx, Lx)

# set up parameters
dt = 400
H = 30.
g = 10
fexpr = Constant(0)
q_0 = 3
alpha = 60
tau = 200
gamma = 5
q_g = 3

parameters=ConvectiveMoistShallowWaterParameters(H=H, g=g, gamma=gamma,
                                                 tau=tau, q_0=q_0, alpha=alpha)

dirname="tracer_sw_gaussian"

output = OutputParameters(dirname=dirname, dumpfreq=1)

diagnostic_fields = [CourantNumber()]

state = State(mesh,
              dt=dt,
              output=output,
              diagnostic_fields = diagnostic_fields,
              parameters=parameters)

moisture_variable = WaterVapour(name="Q", space="DG",
                                 variable_type=TracerVariableType.mixing_ratio,
                                 transport_flag=True,
                                 transport_eqn=TransportEquationType.advective)

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr,
                             active_tracers=[moisture_variable])

# initial conditions
x, y, = SpatialCoordinate(mesh)
u0 = state.fields("u")
D0 = state.fields("D")
Q0 = state.fields("Q_mixing_ratio")
gaussian = 11*exp(-((x-0.5*Lx)**2/2.5e11 + (y-0.5*Ly)**2/2.5e11))
D0.interpolate(Constant(H) + 0.01 * gaussian)
Q0.interpolate(q_g * Constant(1 - 1e-4))

# Add Bouchut condensation forcing
BouchutForcing(eqns, parameters)

# Build time stepper
stepper = Timestepper(state, ((eqns, RK4(state)),))

stepper.run(t=0, tmax=5*dt)
