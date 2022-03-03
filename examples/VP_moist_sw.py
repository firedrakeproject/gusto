from gusto import *
from firedrake import (PeriodicRectangleMesh, conditional, TestFunction,
                       TrialFunction)

# set up mesh
Lx = 10000e3
Ly = 10000e3
delta = 80e3
nx = int(Lx/delta)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction="x")

# set up parameters
dt = 800
tau = 800 
H = 30.
g = 10
f = 2e-11
lamda_r = 1.1e-5
tau_e = 1e6
q_0 = 3
q_g = 3
alpha = 2
gamma = 5
nu_u = 1e4
nu_D = 1e4
nu_Q = 2e4
parameters = MoistShallowWaterParameters(H=H, g=g, gamma=gamma, tau_e=tau_e,
                                         tau=tau, q_0=q_0, q_g=q_g, alpha=alpha,
                                         nu_u=nu_u, nu_D=nu_D, nu_Q=nu_Q,
                                         lamda_r=lamda_r)

dirname="VP_moist_sw"
x, y, = SpatialCoordinate(mesh)

output = OutputParameters(dirname=dirname)

state = State(mesh, dt=dt, output=output, parameters=parameters)

diffusion_options = [
    ("u", DiffusionParameters(kappa=nu_u, mu=10./delta)),
    ("D", DiffusionParameters(kappa=nu_D, mu=10./delta)),
    ("Q", DiffusionParameters(kappa=nu_Q, mu=10./delta))]


eqns = MoistShallowWaterEquations(state, "BDM", 1, fexpr=Constant(f),
                                  diffusion_options=diffusion_options,
                                  no_normal_flow_bc_ids=[1,2])

# initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
Q0 = state.fields("Q")

# spaces
VD = D0.function_space()
E = Function(VD)
C = Function(VD)

