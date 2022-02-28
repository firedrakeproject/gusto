from gusto import *
from firedrake import (PeriodicRectangleMesh, conditional, TestFunction,
                       TrialFunction)

# set up mesh
Lx = 10000e3
Ly = 10000e3
delta_x = 80e3
nx = int(Lx/delta_x)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction="x")

# set up parameters
dt = 800
tau = 800 
H = 1000. # picked this myself
g = 10
f = 2e-11
lamda_r = 1.1e-5
tau_e = 1e6
q_0 = 3
q_g = 3
alpha = 2
gamma = 5
nu_u = 1e4
nu_h = 1e4
nu_q = 2e4
parameters = MoistShallowWaterParameters(H=H, g=g)

dirname="VP_moist_sw"
x, y, = SpatialCoordinate(mesh)

output = OutputParameters(dirname=dirname)

state = State(mesh, dt=dt, output=output, parameters=parameters)

eqns = MoistShallowWaterEquations(state, "BDM", 1, fexpr=Constant(f), no_normal_flow_bc_ids=[1,2])

u0 = state.fields("u")
D0 = state.fields("D")
Q0 = state.fields("Q")

VD = D0.function_space()
E = Function(VD)
C = Function(VD)

Eexpr = (q_g - q) * (q_g - q)/tau_e
Cexpr = (q - q_e) * (q - q_e)/tau

E.interpolate(conditional(q_g > q, Eexpr, 0))
C.interpolate(conditional(q > q_s, Cexpr, 0))
