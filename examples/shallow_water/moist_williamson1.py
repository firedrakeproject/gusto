from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, pi,
                       cos, sin, acos, conditional, VectorFunctionSpace,
                       FiniteElement, exp)
from os import path
from netCDF4 import Dataset

# ---------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------- #

day = 24*60*60
dt = 1000
ref_level = 5
a = 6371220.
R = a/3
u_max = 2*pi*a/(12*day)
alpha = pi/2
h_0 = 100
theta_c = pi
lamda_c = pi/2
tmax = 12*day

# ---------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------- #

# Domain
mesh = IcosahedralSphereMesh(radius=a,
                             refinement_level=ref_level, degree=1)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)
domain = Domain(mesh, dt, 'BDM', 1)
theta, lamda = latlon_coords(mesh)

DG_space = domain.spaces.DG
tracers = [CloudWater(space='tracer')]
eqns = ForcedAdvectionEquation(domain, DG_space, field_name="water_vapour",
                               active_tracers=tracers)

# saturation field (SSHS lat only)
alpha_0 = 0
msat_expr = 110 * exp(-(theta**2/(pi/3)**2) - alpha_0 * (lamda - pi)**2/(2*pi/3)**2)
VD = FunctionSpace(mesh, "DG", 1)
msat = Function(VD)
msat.interpolate(msat_expr)

# I/O
dirname = "moist_williamson1"
output = OutputParameters(dirname=dirname,
                          dumpfreq=1,
                          log_level='INFO')
diagnostic_fields = [CourantNumber(), Sum("water_vapour", "cloud_water")]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

physics_schemes = [(ReversibleAdjustment(eqns, msat, vapour_name='water_vapour',
                                         cloud_name='cloud_water',
                                         set_tau_to_dt=True), RK4(domain))]

# Time stepper
stepper = PrescribedTransport(eqns, RK4(domain), io)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields("u")
v0 = stepper.fields("water_vapour")

u_zonal = u_max * (cos(theta)*cos(alpha) + sin(theta)*cos(lamda)*sin(alpha))
u_merid = -u_max * sin(lamda) * sin(alpha)

u_expr = sphere_to_cartesian(mesh, u_zonal, u_merid)

r = a * (
    acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_c)))

h_expr = h_0/2 * (1 + cos((pi*r)/R))

u0.project(u_expr)
v0.project(conditional(r < R, h_expr, 0))

sat_field = stepper.fields("sat_field", space=VD)
sat_field.interpolate(msat)

# total_moisture = stepper.fields("total_moisture", space=VD)
# total_moisture.interpolate(v0)

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)