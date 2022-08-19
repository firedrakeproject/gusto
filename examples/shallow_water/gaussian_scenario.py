import gusto
import firedrake as fd 
import sys

dt = 0.01
tmax = 2. 
dumpfreq = 1


H = 1.
f = 1.
g = 1.

# Set-up the mesh doubly periodic
n = 20
mesh = fd.PeriodicUnitSquareMesh(n,n)

# set up output parameters
output = gusto.OutputParameters(dirname='gaussian_shallow_water',
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
h0 = state.fields("D")
# u_max not needed for waves setup.
hexpr = fd.exp(-50*((x-0.5)**2 + (y-0.5)**2))
g = parameters.g
h0.interpolate(hexpr+H)
# set up timestepper
stepper = gusto.Timestepper(state, ((eqns, gusto.RK4(state)),))

stepper.run(t=0, tmax=tmax)
