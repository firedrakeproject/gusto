from math import pi
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, cos, sin, acos, conditional,
                       Constant, exp, Function, VectorFunctionSpace,
                       FiniteElement)

SSHS_sat = True
SSHS_lat_only = True

# ---------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------- #

day = 24.*60.*60.
dt = 1000
refinement_level = 3
a = 6371220.
alpha = 0 #pi/2
lamda_c = 3*pi/2
theta_c = 0
R = a/3
u_max = 2*pi*a/(12*day)  # Maximum amplitude of the zonal wind (m/s)
H = 5960
tmax = 12*day
dumpfreq = 23
# initial vapour depends on the choice of saturation field
if SSHS_sat:
    if SSHS_lat_only:
        h_max = 0.9
    else:
        h_max = 0.087
else:
    h_max = 0.032

# ---------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------- #

# Domain
mesh = IcosahedralSphereMesh(radius=a,
                             refinement_level=refinement_level, degree=1)
domain = Domain(mesh, dt, 'BDM', 1)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)
theta, lamda = latlon_coords(mesh)

# Equation
# parameters = ShallowWaterParameters(H=H)
# Omega = parameters.Omega
# fexpr = 2*Omega*x[2]/R

# tracers = [CloudWater(name='cloud', space='DG')]

# eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
#                              active_tracers=tracers)

Vu = VectorFunctionSpace(mesh, "CG", 1)
eltDG = FiniteElement("DG", "interval", 1, variant="equispaced")
VD = FunctionSpace(mesh, eltDG)
tracers = [CloudWater(space='tracer')]
eqns = ForcedAdvectionEquation(domain, VD, field_name="vapour", Vu=Vu,
                               active_tracers=tracers)

# I/O
dirname = 'temp_moist_williamson1_SSHSsat=' + str(bool(SSHS_sat))
output = OutputParameters(dirname=dirname,
                          dumpfreq=1,
                          log_level='INFO')
diagnostic_fields = [CourantNumber()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Physics schemes
# choose saturation profile: SSHS 2021 paper OR varying in latitude
if SSHS_sat:
    alpha_0 = 0
    msat_expr = exp(-(theta**2/(pi/3)**2) - alpha_0 * (lamda - pi)**2/(2*pi/3)**2)
    
else:
    # saturation function which is based on latitude and is unchanging in time
    Gamma = Constant(-37.5604) # lapse rate - this could be lat-dependent
    T0 = Constant(300)   # temperature at equator
    T = Gamma * abs(theta) + T0   # temperature profile
    e1 = Constant(0.98)  # level of saturation
    msat_expr = 3.8e-3 * exp((18 * T - 4824)/(T - 30)) # saturation profile

VD = FunctionSpace(mesh, "DG", 1)
msat = Function(VD)
msat.interpolate(msat_expr)
print(msat.dat.data.max())

physics_schemes = [(ReversibleAdjustment(eqns, msat, vapour_name='vapour',
                                         set_tau_to_dt=True),
                    ForwardEuler(domain))]

# stepper = SplitPhysicsTimestepper(eqns, RK4(domain), io)
stepper = PrescribedTransport(eqns, RK4(domain), io)


# ---------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------- #

u0 = stepper.fields('u')
# D0 = stepper.fields('D')
v0 = stepper.fields('vapour')
# c0 = stepper.fields('cloud')


r = a * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_c))

u_zonal = u_max *(cos(theta)*cos(alpha) + sin(theta)*cos(lamda)*sin(alpha))
u_merid = -u_max*sin(lamda)*sin(alpha)
u_expr = sphere_to_cartesian(mesh, u_zonal, u_merid)

vapour_expr = (h_max/2)*(1 + cos(pi*r/R))

u0.project(u_expr)
v0.interpolate(conditional(r < R, vapour_expr, 0))

# write saturation field out for visualisation
sat_field = stepper.fields("sat_field", space=VD)
sat_field.interpolate(msat)

# ---------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
