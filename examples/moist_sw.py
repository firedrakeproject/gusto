from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, as_vector,
                       sin, cos, acos, exp, sqrt, conditional)
from math import pi
import sys

day = 24.*60.*60.
dt = 3000.

# setup shallow water parameters
R = 6371220.
H = 4000.

# setup input that doesn't change with ref level or dt
parameters = MoistShallowWaterParameters(H=H)

dirname = "moist_sw"
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=4, degree=3)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'), CourantNumber()]

output = OutputParameters(dirname=dirname,
                          dumplist_latlon = ['u_meridional', 'u_zonal',
                                             "Q", "D"])

state = State(mesh,
              dt=dt,
              output=output,
              diagnostic_fields=diagnostic_fields,
              parameters=parameters)

Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
eqns = MoistShallowWaterEquations(state, "BDM", 1, fexpr=fexpr)

# lat lon co-ordinates
theta, lamda = latlon_coords(mesh)

# interpolate initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
Q0 = state.fields("Q")

# parameters for setting up initial conditions
g = parameters.g
theta_c = 20 * pi/180 
lamda_c = 0
r = R * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_c))
r_w = 1000000
h_f = 10
Dexpr = H - h_f * exp(-(r/r_w)**2)
D0.interpolate(Dexpr)

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

Q_sat = 0.9
Q_off = 0.01
Q_min = 0.05
Q_f = 0.0175
Q_background = Function(state.fields("Q").function_space())
Q_background.interpolate(conditional((Q_sat - Q_off) - (cos(theta) + Q_min) < 0,
                            Q_sat - Q_off, cos(theta) + Q_min))
Qexpr = Q_background + Q_f * exp(-(r/r_w)**2)

# interpolate initial conditions 
u0.project(sphere_to_cartesian(mesh, u_z, u_m))
Q0.interpolate(Qexpr)

advected_fields = []
advected_fields.append((ImplicitMidpoint(state, "u")))
advected_fields.append((SSPRK3(state, "D")))
advected_fields.append((SSPRK3(state, "Q")))

stepper = Timestepper(state, ((eqns, SSPRK3(state)),))
stepper.run(t=0, tmax=10*dt)
