from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, cos, sin, acos, exp, sqrt, conditional)
from math import pi

# parameters
R = 6371220.
day = 24.*60.*60.

dt = 3000.

# setup shallow water parameters
R = 6371220.
H = 4000.
parameters = ShallowWaterParameters(H=H)

dirname="sw_vortex"
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=4, degree=3)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'), CourantNumber()]

output = OutputParameters(dirname=dirname,
                          dumplist_latlon = ['u_meridional', 'u_zonal', "D"])

state = State(mesh,
              dt=dt,
              output=output,
              diagnostic_fields=diagnostic_fields,
              parameters=parameters)

Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr)

# lat lon co-ordinates
theta, lamda = latlon_coords(mesh)

# interpolate initial conditions
u0 = state.fields("u")
D0 = state.fields("D")

# parameters for setting up initial conditions
g = parameters.g
theta_c = 20 * pi/180 
lamda_c = 0
r = R * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_c))
r_w = 1000000
h_f = 10
Dexpr = H - h_f * exp(-(r/r_w)**2)

dr_dtheta = (
    (sin(theta_c) * cos(theta)
     - cos(theta_c) * sin(theta) * cos(lamda - lamda_c)) /
    sqrt(1 - (sin(theta_c) * sin(theta) +
              cos(theta_c) * cos(theta) * cos(lamda - lamda_c))**2)
    )
dr_dlamda = (
    cos(theta_c) * cos(theta) * sin(lamda_c - lamda) /
    sqrt(1 - (sin(theta_c) * sin(theta) +
              cos(theta_c) * cos(theta) * cos(lamda - lamda_c))**2)
    )

def uexpr():
    u_zonal = (
        (2 * g * h_f)/(R * r_w**2 * fexpr) * r * exp(-(r/r_w)**2) * dr_dtheta
        )
    u_merid = (
        (-2 * g * h_f)/(R * r_w**2 * fexpr * cos(theta)) * r * exp(-(r/r_w)**2)
        * dr_dlamda
        )
    return u_zonal, u_merid

u_z = Function(D0.function_space(), name="u_z").interpolate(conditional(r<1.5*r_w, uexpr()[0], 0.))
u_m = Function(D0.function_space(), name="u_m").interpolate(conditional(r<1.5*r_w, uexpr()[1], 0.))

# interpolate initial conditions 
u0.project(sphere_to_cartesian(mesh, u_z, u_m))
D0.interpolate(Dexpr)

advected_fields = []
advected_fields.append((ImplicitMidpoint(state, "u")))
advected_fields.append((SSPRK3(state, "D")))

stepper = Timestepper(state, ((eqns, SSPRK3(state)),))
stepper.run(t=0, tmax=10*dt)
