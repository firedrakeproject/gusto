from petsc4py import PETSc
PETSc.Sys.popErrorHandler()

from gusto import *
from firedrake import (PeriodicRectangleMesh, conditional, TestFunction,
                       TrialFunction, exp, Constant, sqrt)

# set up mesh
Lx = 10000e3
Ly = 10000e3
delta = 80e3
nx = int(Lx/delta)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction="x")

# set up parameters
dt = 400
H = 30.
g = 10
fexpr = Constant(0)
nu_u = (1.5e4)/4
nu_D = (1e4)/4

parameters=ShallowWaterParameters(H=H, g=g)

dirname="dry_sw_gaussian"

output = OutputParameters(dirname=dirname, dumpfreq=10)

diagnostic_fields = [CourantNumber()]

state = State(mesh,
              dt=dt,
              output=output,
              diagnostic_fields = diagnostic_fields,
              parameters=parameters)

diffusion_options = [
    ("u", DiffusionParameters(kappa=nu_u, mu=10./delta)),
    ("D", DiffusionParameters(kappa=nu_D, mu=10./delta))]

x, y, = SpatialCoordinate(mesh)
sponge_wall_1 = 300e3
sponge_wall_2 = 9700e3
sponge_expr = 10e-5 * (
    exp(-140*((0.5*Ly-(y-Ly/2))/(Ly)))
    + exp(-140*((y-Ly/2+0.5*Ly)/(Ly))))
sponge_function = conditional(
    y < sponge_wall_2, conditional(
        y > sponge_wall_1, 0, sponge_expr), sponge_expr)

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr,
                             diffusion_options=diffusion_options,
                             sponge=sponge_function,
                             no_normal_flow_bc_ids=[1,2])

# initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
gaussian = 11*exp(-((x-0.5*Lx)**2/2.5e11 + (y-0.5*Ly)**2/2.5e11))
D0.interpolate(Constant(H) - 0.01 * gaussian)

stepper = Timestepper(state, ((eqns, SSPRK3(state)),),)

stepper.run(t=0, tmax=5400*dt) # 25 days
