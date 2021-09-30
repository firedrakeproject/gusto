from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, cos, sin, acos, conditional)
from math import pi

# parameters
R = 6371220.
day = 24.*60.*60.

# set up mesh
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=3)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

# lat lon co-ordinates
theta, lamda = latlon_coords(mesh)

timestepping = TimesteppingParameters(dt=3000.)
dirname = 'williamson1'
fieldlist = ['u', 'D']

output = OutputParameters(dirname=dirname, dumpfreq=23)

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              fieldlist=fieldlist)

# interpolate initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
alpha = 0.
h_max = 1000
lamda_c = 3*pi/2
theta_c = 0
a = R * 3
r = a * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_c))

uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
m1expr = (h_max/2)*(1 + cos(pi*r/R))


u0.project(uexpr)

# set up advected variable in the same space as the height field
VD = D0.function_space()
m1 = state.fields("m1", space=VD)

# initialise m1 as the height field in W1
m1.interpolate(conditional(r < R, m1expr, 0))

m1eqn = AdvectionEquation(state, VD, equation_form="advective")

advected_fields = []
advected_fields.append(("m1", SSPRK3(state, m1, m1eqn)))


# build time stepper
timestepper = AdvectionDiffusion(state, advected_fields)
timestepper.run(t=0, tmax=12*day)
