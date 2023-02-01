from gusto import *
from firedrake import (PeriodicRectangleMesh, exp, cos, conditional,
                       FunctionSpace, Function, tricontourf)


# ----------------------------------------------------------------- #
# Test case parameters
# ----------------------------------------------------------------- #

dt = 0.02
beta = 0.5
L = 2
k = pi/(2*L)
alpha = 0.15
H = 1
g = 1
tmax = 10000
ndumps = 5

# ----------------------------------------------------------------- #
# Set up model objects
# ----------------------------------------------------------------- #

# Domain
Lx = 40
Ly = 16
delta = 0.2
nx = int(Lx/delta)
ny = int(Ly/delta)

mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction='x')
degree = 1
domain = Domain(mesh, dt, "BDM", degree)
x, y = SpatialCoordinate(mesh)


# Equations
params = ShallowWaterParameters(H=H)
fexpr = beta*(y-(Ly/2))
expy = exp(-0.25*(y-(Ly/2))**2)
forcing = cos(k*(x-(Lx/2)))*expy
# forcing = -((y-(Ly/2)) + 1)*(cos(k*(x-(Lx/2)))*expy)
forcing_expr = conditional(x > ((Lx/2) - L), conditional(x < ((Lx/2) + L), forcing, 0), 0)
eqns = LinearShallowWaterEquations(domain, params, fexpr=fexpr,
                                   forcing_expr=forcing_expr,
                                   u_dissipation=alpha, D_dissipation=alpha,
                                   no_normal_flow_bc_ids=[1, 2])

# I/0
dirname = "GV_Gill_heating_Jan20"
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname, dumpfreq=1)
diagnostic_fields = [CourantNumber(), RelativeVorticity()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# timestepper
stepper = Timestepper(eqns, RK4(domain), io)

# ----------------------------------------------------------------- #
# Initial conditions
# ----------------------------------------------------------------- #

u0 = stepper.fields("u")
D0 = stepper.fields("D")

# plot the forcing to check if it is correct
forcing_space = FunctionSpace(domain.mesh, "DG", 1)
forcing_func = Function(forcing_space)
forcing_func.interpolate(forcing_expr)
import matplotlib.pyplot as plt
tricontourf(forcing_func)
plt.xlim(10, 35)
plt.ylim(3, 13)
ax = plt.gca()
PCM=ax.get_children()[2]
plt.colorbar(PCM, ax=ax)
plt.show()

# D0.interpolate(0.1*forcing_expr)

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

stepper.run(t=0, tmax=4*dt)
