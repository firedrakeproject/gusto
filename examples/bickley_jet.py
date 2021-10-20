from gusto import *
from firedrake import PeriodicRectangleMesh
from math import pi

# set up mesh
Lx = 3000e3
Ly = 1000e3
delta_x = 5e3
nx = int(Lx/delta_x)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction='x')

# set up parameters
H = 5960.
angle = 56 * pi/180
parameters = ShallowWaterParameters(H=H)
g = parameters.g
Omega = parameters.Omega
f = 2 * Omega * sin(angle)
dt = 10.

dirname="bickley_jet"
x = SpatialCoordinate(mesh)

output = OutputParameters(dirname=dirname)

state = State(mesh, dt=dt, output=output, parameters=parameters)

eqns = ShallowWaterEquations(state, "BDM", 1)

u0 = state.fields("u")
D0 = state.fields("D")
