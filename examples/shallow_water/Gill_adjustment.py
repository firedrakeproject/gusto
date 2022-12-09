from gusto import *
from firedrake import PeriodicRectangleMesh, exp, Constant

# set up mesh
y_scale = 1
Lx = 80*y_scale
Ly = 8*y_scale
delta = 0.2
nx = int(Lx/delta)
ny = int(Ly/delta)

mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction='x')
x, y = SpatialCoordinate(mesh)

# set up parameters
dt = 0.002
g = 1
beta = 1
H = 1
fexpr = beta*(y-Ly/2)
T = 1/beta*y_scale

parameters = ShallowWaterParameters(H=H, g=g)

dirname = "Gill_adjustment"

output = OutputParameters(dirname=dirname, dumpfreq=1)

diagnostic_fields = [CourantNumber()]

state = State(mesh,
              dt=dt,
              output=output,
              diagnostic_fields=diagnostic_fields,
              parameters=parameters)

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr,
                             no_normal_flow_bc_ids=[1,2])

# initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
epsilon = 0.3
a = 0.3
Dexpr = 1 - epsilon*exp(-((x-30)**2/(2*a**2) + (y-Ly/2)**2/(2*a**2)))
D0.interpolate(Dexpr)

# build timestepper
stepper = Timestepper(eqns, RK4(state), state)

stepper.run(t=0, tmax=5*T)
