from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, FunctionSpace, cos, sin, acos, conditional,
                       Constant)
from math import pi

# shallow water parameters
a = 6371220.
H = 5960.
day = 24.*60.*60.

# set up mesh
mesh = IcosahedralSphereMesh(radius=a,
                                 refinement_level=3, degree=3)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

# spherical co-ordinates
theta, lamda = latlon_coords(mesh)

timestepping = TimesteppingParameters(dt=3000.)
dirname = 'williamson1'
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)

output = OutputParameters(dirname=dirname, steady_state_error_fields=['D', 'u'])

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              fieldlist=fieldlist)

# interpolate initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
u_max = 2*pi*a/(12*day)
alpha = 0.
h_max = 1000
lamda_c = 3*pi/2
theta_c = 0
R = a/3
r = a * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_c))
uexpr = as_vector(
    [u_max*(cos(theta)*cos(alpha) + sin(theta)*cos(lamda)*sin(alpha)),
     -u_max*sin(lamda)*sin(alpha), 0.0])
Dexpr = (h_max/2)*(1 + cos(pi*r/R))

V = FunctionSpace(mesh, "CG", 1)
f = state.fields("coriolis", V) # think it is looking for Coriolis
f.interpolate(Constant(0))

u0.project(uexpr)
D0.interpolate(conditional(r < R, Dexpr, 0))
state.initialise([('u', u0),
                  ('D', D0)])


### I took this part from W2 case code and I don't really know what's going on
Deqn = AdvectionEquation(state, D0.function_space(), equation_form="advective")

advected_fields = []
advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

# build time stepper
timestepper = AdvectionDiffusion(state, advected_fields)

timestepper.run(t=0, tmax=6000)
