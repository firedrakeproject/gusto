from gusto import *
from firedrake import (PeriodicRectangleMesh, exp, Constant, sqrt, cos,
                       conditional, FunctionSpace, Function)

# set up mesh
Lx = 40
Ly = 16
delta = 0.2
nx = int(Lx/delta)
ny = int(Ly/delta)

mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction='x')
x, y = SpatialCoordinate(mesh)

# set up parameters
dt = 0.02
beta = 0.5
L = 2
k = pi/(2*L)
alpha = 0.15
H = 1
g = 1
tmax = 10000
fexpr = beta*(y-(Ly/2))

dirname = "GV_Gill_heating"

parameters = ShallowWaterParameters(H=H)

output = OutputParameters(dirname=dirname, dumpfreq=1)

diagnostic_fields = [CourantNumber(), RelativeVorticity()]

state = State(mesh,
              dt=dt,
              output=output,
              diagnostic_fields=diagnostic_fields,
              parameters=parameters)

expy = exp(-0.25*(y-(Ly/2))**2)

# forcing = cos(k*(x-(Lx/2)))*exp(0.25*(y-(Ly/2))**2)
forcing = -((y-(Ly/2)) + 1)*(cos(k*(x-(Lx/2)))*expy)
forcing_expr = conditional(x>((Lx/2)-L), conditional(x<((Lx/2)+L), forcing, 0), 0)

alpha = Constant(0.15)
eqns = LinearShallowWaterEquations(state, "BDM", 1, fexpr=fexpr,
                                   forcing_expr=forcing_expr,
                                   u_dissipation=alpha, D_dissipation=alpha,
                                   no_normal_flow_bc_ids=[1,2])

# initial conditions
u0 = state.fields("u")
D0 = state.fields("D")

# D0.interpolate(0.1*forcing_expr)

# timestepper
stepper = Timestepper(eqns, ForwardEuler(state), state)

stepper.run(t=0, tmax = 4*dt)
