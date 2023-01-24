"""
An implementation of the Williams 3 Test case 
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi, exp, cos
import sys

# Set up timestepping variables 
day = 24. * 60. *60.
ref = 3
dt = 4000
tmax = 5*day
ndumps = 5

# Shallow Water Parameters
a = 6371220.
H = 5960.

parameters = ShallowWaterParameters(H=H)

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Mesh and domain
mesh = IcosahedralSphereMesh(a, ref=ref, degree=1)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)
domain = Domain(mesh, dt, "BDM", 1)

#Equations
lat, lon = latlon_coords(mesh)
Omega = parameters.Omega
fexpr = 2*Omega * sin(lat)
eqns = ShallowWaterEquations(domain,parameters,fexpr=fexpr, u_transport_option='vector_advection_form')

#Output and IO
dirname = "williamson_2_ref%s_dt%s" % (ref, dt)
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist_latlon=['D', 'D_error'],
                          log_level='INFO')
diagnostic_fields = [CourantNumber,SteadyStateError('u'), SteadyStateError('D') ]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport Fields and time stepper
transported_fields = [ImplicitMidpoint(domain, "u"),
                     SSPRK3(domain, "D", subcycles=2)]
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields)

# ------------------------------------------------------------------------ #
# Initial Conditions
# ------------------------------------------------------------------------ #

def b_func(x):
    b = conditional(x<=0, 0, exp(-x**-1) )
    return b

u0 = stepper.fields('u')
D0 = stepper.fields('D')
xe = 0.3
u_0 = 2 * pi * a / (12*day)
lat_b = -pi / 6
lat_e = pi / 2
g = parameters.g
h0 = 2.94e4 / g

x_mod = xe*(lat - lat_b )/(lat_e - lat_b)
u_expr = as_vector([u_0 * b_func(x_mod) * b_func(x_mod - xe) * exp(4 / xe),0])
h_expr = h0 + 2*a*Omega/g * cos(lat)


u0.project(u_expr)
D0.interpolate(h_expr)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# Run!
stepper.run(t=0, tmax=tmax)