"""
Rossby-Haurwitz wave
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi, exp, cos
import numpy as np

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
mesh = IcosahedralSphereMesh(radius=a,
                            refinement_level=ref, degree=1)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)
domain = Domain(mesh, dt, "BDM", 1)

#Equations
lat, lon = latlon_coords(mesh)
Omega = parameters.Omega
fexpr = 2*Omega * x[2] / a
eqns = ShallowWaterEquations(domain,parameters,fexpr=fexpr, u_transport_option='vector_advection_form')

#Output and IO
dirname = 'Rossby-Haurwitx_Wave'  
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist_latlon=['D', 'D_error'],
                          log_level='INFO')
diagnostic_fields = [CourantNumber(),SteadyStateError('u'), SteadyStateError('D'),RelativeVorticity() ]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport Fields and time stepper
transported_fields = [SSPRK3(domain, "u"),
                     SSPRK3(domain, "D")]

stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields)

# ------------------------------------------------------------------------ #
# Initial Conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')

Vu = domain.spaces("HDiv")
K = 7 # need to find constant
R = pi / 9 # double check

phi = -a**2 * Omega * sin(lat) + a**2 * K * cos(lat)**R * sin(lat) * cos(R*lat)

u_expr = Gradient(phi)



