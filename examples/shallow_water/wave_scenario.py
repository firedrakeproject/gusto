import gusto
import firedrake as fd 
import sys

dt = 1.
tmax = 10.
dumpfreq = 1


H = 1.
f = 1.
g = 1.

# Set-up the mesh doubly periodic
n = 20
mesh = fd.PeriodicUnitSquareMesh(n,n)

# set up output parameters
output = gusto.OutputParameters(dirname='wave_shallow_water',
                          dumpfreq=dumpfreq,
                          log_level='INFO')

# set up physical parameters for the shallow water equations
parameters = gusto.ShallowWaterParameters(H=H, g=g)

# set up state
state = gusto.State(mesh,
              dt=dt,
              output=output,
              parameters=parameters)

# set up equations
x,y = fd.SpatialCoordinate(mesh)
eqns = gusto.LinearShallowWaterEquations(state, "BDM", 1, fexpr=fd.Constant(1.))

# interpolate initial conditions
# Initial/current conditions
u0 = state.fields("u")
h0 = state.fields("D")
# u_max not needed for waves setup.
uexpr = fd.as_vector([fd.cos(8*fd.pi*x)*fd.cos(2*fd.pi*y), fd.cos(4*fd.pi*x)*fd.cos(4*fd.pi*y)])
hexpr = fd.sin(4*fd.pi*x)*fd.cos(2*fd.pi*y) - 0.2*fd.cos(4*fd.pi*x)*fd.sin(4*fd.pi*y)
g = parameters.g
u0.project(uexpr)
h0.interpolate(hexpr+H)
# set up timestepper
stepper = gusto.Timestepper(state, ((eqns, gusto.RK4(state)),))

stepper.run(t=0, tmax=tmax)
