from firedrake import (ExtrudedMesh,
                       SpatialCoordinate, cos, sin, pi, sqrt,
                       exp, Constant, Function, as_vector,
                       FunctionSpace, VectorFunctionSpace,
                       errornorm, norm, min_value, max_value)
from gusto import *
from gusto.diagnostics import SolidBodyImbalance, GeostrophicImbalance                                              # 
# ---------------------------------------------------------------------------- #
# Test case Parameters
# ---------------------------------------------------------------------------- #
dt = 1000.
days = 30.
ndumps = 60
tmax = days * 24. * 60. * 60.
deltaz = 2.0e3
dumpfreq = int(24. * 60. * 60. / (2 * dt)) # dumps twice a day 
# -----------------------------------------------------------------------------#
# Set up Model
# ---------------------------------------------------------------------------- #

# Domain
a = 6.371229e6  # radius of earth
Height = 3.0e4  # height
nlayers = int(Height/deltaz)

m = GeneralCubedSphereMesh(a, num_cells_per_edge_of_panel=24, degree=2)
mesh = ExtrudedMesh(m, layers=nlayers, 
                    layer_height=Height/nlayers, 
                    extrusion_type='radial')
domain = Domain(mesh, dt, "RTCF", degree=1)

# Equations
params = CompressibleParameters()
omega = Constant(7.292e-5)
Omega = as_vector((0, 0, omega))

eqn = CompressibleEulerEquations(domain, params, Omega=Omega, 
                                 u_transport_option='vector_invariant_form')

dirname = 'SBR_meshdeg2_order2_dt1000_vectorInvar'
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dump_nc=True,
                          dump_vtus=False,
                          log_level=('INFO'))
diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'), 
                     RadialComponent('u'), CourantNumber(), 
                     HydrostaticImbalance(eqn)]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport Schemes
transported_fields = []
transported_fields.append(ImplicitMidpoint(domain, "u"))
transported_fields.append(SSPRK3(domain, "rho"))
transported_fields.append(SSPRK3(domain, "theta", options=SUPGOptions()))

transport_methods = [DGUpwind(eqn, field) for field in ["u", "rho", "theta"]]

# Linear Solver
linear_solver = CompressibleSolver(eqn)

# Time Stepper
stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver)

# -----------------------------------------------------------------------------#
# Initial Conditions
# -----------------------------------------------------------------------------#

x, y, z = SpatialCoordinate(mesh)
lat, lon = latlon_coords(mesh)
r = sqrt(x**2 + y**2 + z**2)
l = sqrt(x**2 + y**2)
unsafe_x = x / l
unsafe_y = y / l
safe_x = min_value(max_value(unsafe_x, -1), 1)
safe_y = min_value(max_value(unsafe_y, -1), 1)

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
q_expr = Q_expr + (a - r) * g * a / (Rd * T0 * r)
p_expr = p0 * exp(q_expr)
theta_expr = T0 * (p_expr / p0) ** (-params.kappa)
pie_expr = T0 / theta_expr
rho_expr = p_expr / (Rd * T0)

# get components of u in spherical polar coordinates
zonal_u = u0 * r / a * cos(lat)
merid_u = Constant(0.0)
radial_u = Constant(0.0)

# now convert to global Cartesian coordinates
u_x_expr = zonal_u * -safe_y
u_y_expr = zonal_u * safe_x
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
compressible_hydrostatic_balance(eqn, theta0, rho0, 
                                 exner_boundary=pie, 
                                 solve_for_rho=True)
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
print('Initilise Windy Boi')
stepper.run(t=0, tmax=tmax)
