from gusto import *
from firedrake import PeriodicRectangleMesh, UnitSquareMesh
from math import pi, sin, cos
from mpmath import sech

# set up mesh
Lx = 3000
Ly = 1000
delta_x = 5
nx = int(Lx/delta_x)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction='x')

# set up parameters
H = 5960. # making this choice
angle = 60 * pi/180 # making this choice
parameters = ShallowWaterParameters(H=H)
g = parameters.g
Omega = parameters.Omega
f = 2 * Omega * sin(angle)
L = 6
dh = 59.6
dt = 10.

dirname="bickley_jet"
x, y = SpatialCoordinate(mesh)

output = OutputParameters(dirname=dirname)

state = State(mesh, dt=dt, output=output, parameters=parameters)

eqns = ShallowWaterEquations(state, "BDM", 1)

u0 = state.fields("u")
D0 = state.fields("D")

uexpr = Constant(0)
vexpr = (-g*dh/f*L) * sech(x/L)**2
#test = -0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)
