from gusto import *
from firedrake import (PeriodicRectangleMesh, exp, Constant, sqrt, cos,
                       conditional)

# mesh depends on parameters so set these up first
H = 400
beta = 2.3e-11
parameters = ShallowWaterParameters(H=H)
g = parameters.g
c = sqrt(g*H)
Req = sqrt(c/(2*beta))

# set up mesh
Lx = 20*Req
Ly = 10*Req
delta = 0.1*Req
nx = int(Lx/delta)
ny = int(Ly/delta)

mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction='x')
x, y = SpatialCoordinate(mesh)

# set up the rest of the parameters
dt = 200
fexpr = beta*(y-(Ly/2))
T = 1/sqrt(2*beta*c)

dirname = "Gill_heating"

output = OutputParameters(dirname=dirname, dumpfreq=1)

diagnostic_fields = [CourantNumber(), RelativeVorticity()]

state = State(mesh,
              dt=dt,
              output=output,
              diagnostic_fields=diagnostic_fields,
              parameters=parameters)

eqns = LinearShallowWaterEquations(state, "BDM", 1, fexpr=fexpr,
                             no_normal_flow_bc_ids=[1,2])

# initial conditions
u0 = state.fields("u")
D0 = state.fields("D")

L = 2*Req  # radius of perturbed region
k = 2*pi/L
forcing = cos(k*x)*exp(0.25*y**2)
forcing_expr = conditional(x>-L, conditional(x<L, forcing, 0), 0)
