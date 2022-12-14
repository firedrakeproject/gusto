from gusto import *
from firedrake import (CubedSphereMesh, ExtrudedMesh,
                       SpatialCoordinate, cos, sin, pi, sqrt,
                       exp, Constant, Function, as_vector,
                       FunctionSpace, VectorFunctionSpace,
                       errornorm, norm, Min, Max)

dt = 1800
days = 12
ndumps = 12
tmax = days * 24 * 60 * 60
deltaz = 2.0e3

# make mesh
a = 6.371229e6  # radius of earth
Height = 3.0e4  # height
nlayers = int(Height/deltaz)
ref_level = 3
m = CubedSphereMesh(radius=a, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Height/nlayers, extrusion_type='radial')

x, y, z = SpatialCoordinate(mesh)
lat, lon = latlon_coords(mesh)

r = sqrt(x**2 + y**2 + z**2)
l = sqrt(x**2 + y**2)
unsafe_xl = x/l
safe_xl = Min(Max(unsafe_xl, -1.0), 1.0)
unsafe_yl = y/l
safe_yl = Min(Max(unsafe_yl, -1.0), 1.0)


# options
dirname = 'sbr_quadratic_%i_day_dt_%i_degree%i' % (days, dt, 2)

output = OutputParameters(dirname=dirname,
                          dumpfreq=int(int(dt * 5)),
                          dumplist=['u', 'rho', 'theta'],
                          perturbation_fields=['theta'],
                          dumplist_latlon=['u_meridional',
                                           'u_zonal',
                                           'u_radial'],
                          log_level='INFO')

# set up parameters
params = CompressibleParameters()
omega = Constant(7.292e-5)
phi0 = Constant(pi/4)
Rd = params.R_d
cp = params.cp
f0 = 2 * omega * sin(phi0)
Omega = as_vector((0, 0, f0))
g = params.g
p0 = Constant(100000)
T0 = 280.  # in K
u0 = 40.

diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'), RadialComponent('u'), CourantNumber()]
state = State(mesh,
              dt=dt,
              output=output,
              parameters=params,
              diagnostic_fields=diagnostic_fields)

eqns = CompressibleEulerEquations(state, "RTCF", 1, Omega=Omega, u_transport_option='vector_advection_form')

# Initial conditions
u = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

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
theta_expr = T0 * p_expr ** (-params.kappa) / p0
pie_expr = T0 / theta_expr
rho_expr = rho(params, theta_expr, pie_expr)

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
compressible_hydrostatic_balance(state, theta0, rho0, exner_boundary=pie, solve_for_rho=False)

print('make analytic rho')
rho_analytic = Function(Vr).interpolate(rho_expr)
print('Normalised rho error is:', errornorm(rho_analytic, rho0) / norm(rho_analytic))

# make mean fields
print('make mean fields')
rho_b = Function(Vr).assign(rho0)
theta_b = Function(Vt).assign(theta0)

# assign reference profiles
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up transport schemes
transported_fields = []
transported_fields.append(ImplicitMidpoint(state, "u"))
transported_fields.append(SSPRK3(state, "rho"))
transported_fields.append(SSPRK3(state, "theta", options=SUPGOptions()))

# Set up linear solver
linear_solver = CompressibleSolver(state, eqns)

# build time stepper
stepper = SemiImplicitQuasiNewton(eqns, state, transported_fields,
                                  linear_solver=linear_solver)

stepper.run(t=0, tmax=tmax)
