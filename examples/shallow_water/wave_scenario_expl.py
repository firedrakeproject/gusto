from firedrake import PeriodicUnitSquareMesh, Constant, SpatialCoordinate, \
    cos, sin, pi
from gusto.rexi import *
from gusto import *

dt = 0.1
tmax = 0.1

# Set-up the mesh doubly periodic
n = 20
mesh = PeriodicUnitSquareMesh(n, n)
domain = Domain(mesh, dt, "BDM", 1)

# set up shallow water equations
H = 1.
f = 1.
g = 1.
parameters = ShallowWaterParameters(H=H, g=g)
eqns = LinearShallowWaterEquations(domain, parameters, fexpr=Constant(f))

# I/O
output = OutputParameters(dirname='waves_shallow_water',
                          dumpfreq=1)
io = IO(domain, output)

# set up timestepper
stepper = Timestepper(eqns, ExponentialEuler(domain, Rexi(eqns, RexiParameters())), io)

# interpolate initial conditions
x, y = SpatialCoordinate(mesh)
u0 = stepper.fields("u")
h0 = stepper.fields("D")
uexpr = as_vector([cos(8*pi*x)*cos(2*pi*y), cos(4*pi*x)*cos(4*pi*y)])
hexpr = sin(4*pi*x)*cos(2*pi*y) - 0.2*cos(4*pi*x)*sin(4*pi*y)
u0.project(uexpr)
h0.interpolate(hexpr)

stepper.run(t=0, tmax=tmax)
