from firedrake import (IcosahedralSphereMesh, PeriodicIntervalMesh,
                       ExtrudedMesh, SpatialCoordinate, as_vector,
                       sin, exp, pi)

from gusto import *

mesh = IcosahedralSphereMesh(radius=1,
                             refinement_level=3,
                             degree=1)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

dirname="advection_RK4"

dt = pi/3. * 0.002
output = OutputParameters(dirname=dirname, dumpfreq=10)
diagnostic_fields = [CourantNumber()]
state = State(mesh, dt=dt, output=output, diagnostic_fields=diagnostic_fields)

uexpr = as_vector([-x[1], x[0], 0.0])

tmax = pi
f_init = exp(-x[2]**2 - x[0]**2)

V = state.spaces("DG", "DG", 1)
eqn = AdvectionEquation(state, V, "f", "BDM", 1)

state.fields("f").interpolate(f_init)
state.fields("u").project(uexpr)

advection_scheme = [(eqn, RK4(state))]
timestepper = PrescribedAdvection(state, advection_scheme)
timestepper.run(0, tmax)
