from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, as_vector,
                       FunctionSpace, exp)
from math import pi

# radius of sphere
R = 6371220.

dirname = "sdc_advection"
mesh = IcosahedralSphereMesh(radius=R, refinement_level=4, degree=3)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

output = OutputParameters(dirname=dirname)

dt = 300
state = State(mesh,
              dt=dt,
              output=output)

Vdg = FunctionSpace(mesh, "DG", 1)

eqn = AdvectionEquation(state, Vdg, "tracer", "BDM", 1)
eqn.residual = advecting_velocity.remove(eqn.residual)
for t in eqn.residual:
    print(t.labels.keys())


# interpolate initial conditions
u0 = state.fields("u")
day = 24.*60.*60.
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
u0.project(uexpr)

q0 = state.fields("tracer")
q0.interpolate(exp(-x[2]**2 - x[0]**2))

M = 3
maxk = 2
scheme = IMEX_SDC(state, M, maxk)
timestepper = Timestepper(state, ((eqn, scheme),))
tmax = 12*day
timestepper.run(0, tmax)
