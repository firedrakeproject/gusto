from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, FunctionSpace, cos, sin, acos, conditional,
                       Constant)
from math import pi 

nondivergent = True

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
dirname = 'deformational_advection'
fieldlist = ['u', 'D']

output = OutputParameters(dirname=dirname, dumpfreq=23)

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              fieldlist=fieldlist)

# set up initial conditions
u0 = state.fields("u")
D0 = state.fields("D")

# parameters for initial conditions from Lauritzen
u_max = 2*pi*R/(12*day) 
alpha = 0.
h_max = 1
b = 0.1
c = 0.9
lamda_1 = 3*pi/2 - pi/6
lamda_2 = 3*pi/2 + pi/6 
theta_c = 0
br = R/2
r1 = R * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_1))
r2 = R * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_2))
h1expr = b + c * (h_max/2)*(1 + cos(pi*r1/br))
h2expr = b + c * (h_max/2)*(1 + cos(pi*r2/br))

# velocity parameters
T = 12*day

if nondivergent:
    def uexpr(t):
        lamda_p = lamda - 2*pi*t/T
        u_zonal = (
            10*R/T * (sin(lamda_p))**2 * sin(2*theta) * cos(pi*t/T)
            + 2*pi*R/T * cos(theta)
        )
        u_merid = 10*R/T * sin(2*lamda_p) * cos(theta) * cos(pi*t/T)
        return sphere_to_cartesian(mesh, u_zonal, u_merid)
else:
    def uexpr(t):
        lamda_p = lamda - 2*pi*t/T
        return as_vector(
            [-5*R/T * (sin(lamda_p/2))**2 * sin(2*theta) *
             (cos(theta))**2 * cos(pi*t/T) + 2*pi*R/T * cos(theta),
             5/2*R/T * sin(lamda_p) * (cos(theta))**3 * cos(pi*t/T),
             0.0])

u0.project(uexpr(0))

# set up advected variable in the same space as the height field 
VD = D0.function_space()
m1 = state.fields("m1", space=VD)

# initialise m1 as the two cosine bells
m1.interpolate(conditional(r1 < br, h1expr, conditional(r2 < br, h2expr, b)))

m1eqn = AdvectionEquation(state, VD, equation_form="advective")

advected_fields = []
advected_fields.append(("m1", SSPRK3(state, m1, m1eqn)))

# build time stepper
timestepper = AdvectionDiffusion(state, advected_fields, prescribed_fields=[("u", uexpr)])
timestepper.run(t=0, tmax=12*day)
