from firedrake import (CubedSphereMesh, ExtrudedMesh,
                       SpatialCoordinate, cos, sin, pi, sqrt,
                       exp, Constant, Function, as_vector,
                       FunctionSpace, VectorFunctionSpace,
                       errornorm, norm)
from gusto import *                                              # 
# -------------------------------------------------------------- #
# Test case Parameters
# -------------------------------------------------------------- #
dt = 1800.
days = 30.
ndumps = 60
tmax = days * 24. * 60. * 60.
deltaz = 1.0e3

# -------------------------------------------------------------- #
# Set up Model
# -------------------------------------------------------------- #

# Domain
a = 6.371229e6  # radius of earth
Height = 3.0e4  # height
nlayers = int(Height/deltaz)
ref_level = 4
m = CubedSphereMesh(radius=a, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Height/nlayers, extrusion_type='radial')
domain = Domain(mesh, dt, "RTCF", degree=1)

# Equations
params = CompressibleParameters()
omega = Constant(7.292e-5)
phi0 = Constant(pi/4)
f0 = 2 * omega * sin(phi0)
Omega = as_vector((0, 0, f0))

eqn = CompressibleEulerEquations(domain, params, Omega=Omega, u_transport_option='vector_advection_form')

#dirname = 'sbr_quadratic_%i_day_dt_%i_degree%i_solveforrho' % (days, dt, 2)
dirname = 'solidbody_ref4_layers30'
output = OutputParameters(dirname=dirname,
                          dumpfreq=1,
                          dumplist=['u', 'rho', 'theta'],
                          dumplist_latlon=['u_meridional',
                                           'u_zonal',
                                           'u_radial',
                                           'rho',
                                           'theta'],
                          log_level=('INFO'))
diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'), RadialComponent('u'), CourantNumber()]# CompressibleKineticEnergy()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport Schemes
transported_fields = []
transported_fields.append(ImplicitMidpoint(domain, "u"))
transported_fields.append(SSPRK3(domain, "rho"))
transported_fields.append(SSPRK3(domain, "theta", options=SUPGOptions()))

# Linear Solver
linear_solver = CompressibleSolver(eqn)

# Time Stepper
stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                  linear_solver=linear_solver)

# -------------------------------------------------------------- #
# Initial Conditions
# -------------------------------------------------------------- #

x, y, z = SpatialCoordinate(mesh)
lat, lon = latlon_coords(mesh)
r = sqrt(x**2 + y**2 + z**2)
l = sqrt(x**2 + y**2)
unsafe_xl = x/l
safe_xl = Min(Max(unsafe_xl, -1.0), 1.0)
unsafe_yl = y/l
safe_yl = Min(Max(unsafe_yl, -1.0), 1.0)

# set up parameters
Rd = params.R_d
cp = params.cp
g = params.g
p0 = Constant(100000)
T0 = 280.  # in K
u0 = 40.

u = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# spaces
Vu = u.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()
Vpsi = FunctionSpace(mesh, "CG", 2)
Vec_psi = VectorFunctionSpace(mesh, "CG", 2)

# expressions for variables from paper
s = (r / a) * cos(lat)
Q_expr = s**2 * (0.5 * u0**2 + omega * a * u0) / (Rd * T0)
# solving fields as per the staniforth paper
q_expr = Q_expr - ((g * a**2) / (Rd * T0)) * (a**-1 - r**-1)
p_expr = p0 * exp(q_expr)
theta_expr = T0 * (p_expr / p0) ** (-params.kappa)
pie_expr = T0 / theta_expr
rho_expr = p_expr / (Rd * T0)

# get components of u in spherical polar coordinates
zonal_u = u0 * r / a * cos(lat)
merid_u = Constant(0.0)
radial_u = Constant(0.0)

# now convert to global Cartesian coordinates
u_x_expr = zonal_u * -safe_yl
u_y_expr = zonal_u * safe_xl
u_z_expr = Constant(0.0)

# obtain initial conditions
print('Set up initial conditions')
print('project u')
u.project(as_vector([u_x_expr, u_y_expr, u_z_expr]))
print('interpolate theta')
theta0.interpolate(theta_expr)
print('find pi')
pie = Function(Vr).interpolate(pie_expr)
print('find rho')
rho0.interpolate(rho_expr)
compressible_hydrostatic_balance(eqn, theta0, rho0, exner_boundary=pie, solve_for_rho=True)

print('make analytic rho')
rho_analytic = Function(Vr).interpolate(rho_expr)
print('Normalised rho error is:', errornorm(rho_analytic, rho0) / norm(rho_analytic))

# make mean fields
print('make mean fields')
rho_b = Function(Vr).assign(rho0)
theta_b = Function(Vt).assign(theta0)

# assign reference profiles
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])
stepper.run(t=0, tmax=tmax)
