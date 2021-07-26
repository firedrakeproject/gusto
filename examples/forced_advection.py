from gusto import *
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh,
                       ExtrudedMesh, cos, sin, pi, exp, Function)

# Implement forced advection test from Zerroukat and Allen 2020

nlayers = 50  # horizontal layers
columns = 200  # number of columns
L = 200e3
m = PeriodicIntervalMesh(columns, L)

# build volume mesh
H = 15e3  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

timestepping = TimesteppingParameters(dt=10)
dirname = 'forced_advection'
output = OutputParameters(dirname=dirname)
fieldlist = ["u", "rho", "theta"]

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              fieldlist=fieldlist)

# Initial conditions
u0 = state.fields("u")

# set up uexpr here
x, z = SpatialCoordinate(mesh)
u_mean = 10
uexpr = ((u_mean/2) * (2 + cos(2 * pi * x/L) * cos(pi * z) * sin(pi * z/H)),
         u_mean * H/L * sin(2 * pi * x/L) * sin(pi * z/H)**2)
u0.project(as_vector(uexpr))

# set up moisture variables here
theta0 = state.fields("theta")
Vth = theta0.function_space()
m1 = state.fields("m1", space=Vth)
m2 = state.fields("m2", space=Vth)
m3 = state.fields("m3", space=Vth)

m2.interpolate(H/2 * exp(-((x - L/10)**2 + (z - H/2)**2)/2 * L/5**2))

m1eqn = AdvectionEquation(state, Vth, equation_form="advective")
m2eqn = AdvectionEquation(state, Vth, equation_form="advective")
m3eqn = AdvectionEquation(state, Vth, equation_form="advective")

advected_fields = []
advected_fields.append(("m1", SSPRK3(state, m1, m1eqn)))
advected_fields.append(("m2", SSPRK3(state, m2, m2eqn)))
advected_fields.append(("m3", SSPRK3(state, m3, m3eqn)))

timestepper = AdvectionDiffusion(state, advected_fields)
timestepper.run(t=0, tmax=2000)
