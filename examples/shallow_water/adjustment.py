from gusto import *
from firedrake import PeriodicRectangleMesh, exp, Constant

# set up mesh
# Lx = 200
# Ly = 20
# nx = ny = 1000
Lx = 200
Ly = 20
delta = 1
nx = int(Lx/delta)
ny = int(Ly/delta)

mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly)
x, y = SpatialCoordinate(mesh)

# set up parameters
dt = 0.01
H = 1.
g = 10
beta = 1 #2.286e-11
fexpr = beta*(y-Ly/2)

parameters = ShallowWaterParameters(H=H, g=g)

dirname = "height_adjustment_gaussian_with_rotation"

output = OutputParameters(dirname=dirname, dumpfreq=10)

diagnostic_fields = [CourantNumber()]

state = State(mesh,
              dt=dt,
              output=output,
              diagnostic_fields=diagnostic_fields,
              parameters=parameters)

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr, thermal=True)

# initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
b0 = state.fields("b")
epsilon = 0.2
a = 5
Dexpr = 1 - epsilon*exp(-((x-Lx/2)**2/(2*a**2) + (y-Ly/2)**2/(2*a**2)))
D0.interpolate(Dexpr)
b0.interpolate(Constant(1))

# Build time stepper
stepper = Timestepper(eqns, RK4(state), state)

stepper.run(t=0, tmax=20)


