from firedrake import (ExtrudedMesh,
                       SpatialCoordinate, cos, sin, pi, sqrt, File,
                       exp, Constant, Function, as_vector, acos,
                       errornorm, norm, min_value, max_value, le, ge)
from gusto import *
from gusto.diagnostics import SolidBodyImbalance, GeostrophicImbalance                                              # 
# -------------------------------------------------------------- #
# Test case Parameters
# -------------------------------------------------------------- #
dt = 1000.
days = 10.
ndumps = 60
tmax = days * 24. * 60. * 60.
deltaz = 3.0e3

# -------------------------------------------------------------- #
# Set up Model
# -------------------------------------------------------------- #

# Domain
a = 6.371229e6  # radius of earth
Height = 3.0e4  # height
nlayers = int(Height/deltaz)

m = GeneralCubedSphereMesh(a, num_cells_per_edge_of_panel=12, degree=2)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Height/nlayers, extrusion_type='radial')
domain = Domain(mesh, dt, "RTCF", degree=1)

# Equations
params = CompressibleParameters()
omega = Constant(7.292e-5)
Omega = as_vector((0, 0, omega))

eqn = CompressibleEulerEquations(domain, params, Omega=Omega, u_transport_option='vector_invariant_form')

dirname = 'Baroclinic_pertubationtest'
output = OutputParameters(dirname=dirname,
                          dumpfreq=1, 
                          dumplist=['u', 'rho', 'theta'],
                          dumplist_latlon=['u_meridional',
                                           'u_zonal',
                                           'u_radial',
                                           'rho',
                                           'theta'],
                          log_level=('INFO'))
diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'), RadialComponent('u'), CourantNumber()]
                     
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
Vr = rho0.function_space()
Vt = theta0.function_space()


#TODO We are just using an isotherm here, should we consider a more complex temperature scenario for baroclinic? 

# expressions for variables from paper
s = (r / a) * cos(lat)
Q_expr = s**2 * (0.5 * u0**2 + omega * a * u0) / (Rd * T0)
# solving fields as per the staniforth paper
q_expr = Q_expr + (a - r) * g * a / (Rd * T0 * r)
p_expr = p0 * exp(q_expr)
theta_expr = T0 * (p_expr / p0) ** (-params.kappa) 
pie_expr = T0 / theta_expr
rho_expr = p_expr / (Rd * T0)

# -------------------------------------------------------------- #
# Perturbation
# -------------------------------------------------------------- #

zt = 1.5e4     # top of perturbation
d0 = a / 6     # horizontal radius of perturbation
Vp = 1         # Perturbed wind amplitude  
lon_c , lat_c = pi/9,  2*pi/9 # location of perturbation centre   
err_tol = 1e-12

d = a * acos(sin(lat_c)*sin(lat) + cos(lat_c)*cos(lat)*cos(lon - lon_c)) # distance from centre of perturbation

depth = r - a # The distance from origin subtracted from earth radius
#zeta = conditional(ge(depth,zt-err_tol), 0, 1 - 3*(depth / zt)**2 + 2*(depth / zt)**3) # peturbation vertical taper
zeta = 1 - 3*(depth / zt)**2 + 2*(depth / zt)**3

perturb_magnitude = (16*Vp/(3*sqrt(3))) * zeta * sin((pi * d) / (2 * d0)) * cos((pi * d) / (2 * d0)) ** 3


zonal_pert = conditional(le(d,err_tol), 0, 
                         conditional(ge(d,(d0-err_tol)), 0, -perturb_magnitude * (-sin(lat_c)*cos(lat) + cos(lat_c)*sin(lat)*cos(lon - lon_c)) / sin(d / a)))
meridional_pert = conditional(le(d,err_tol), 0, 
                              conditional(ge(d,d0-err_tol), 0, perturb_magnitude * cos(lat_c)*sin(lon - lon_c) / sin(d / a)))

conditional_test = conditional(le(d,err_tol), 0, 
                         conditional(ge(d,(d0-err_tol)), 0, 10))

testput = File('results/testout.pvd')
d_field = Function(Vr).interpolate(d)
z_field = Function(Vr).interpolate(zeta)
zp_field = Function(Vr).interpolate(zonal_pert)
mp_field = Function(Vr).interpolate(meridional_pert)
condcheck = Function(Vr).interpolate(conditional_test)
testput.write(d_field, z_field, zp_field, mp_field, condcheck)

pertput = File('results/pertout.pvd')
magnitude = Function(Vr).interpolate(perturb_magnitude)
zonal_localistaion = Function(Vr).interpolate((-sin(lat_c)*cos(lat) + cos(lat_c)*sin(lat)*cos(lon - lon_c)) / sin(d / a))
meridional_localisation = Function(Vr).interpolate(cos(lat_c)*sin(lon - lon_c) / sin(d / a))
pertput.write(magnitude, zonal_localistaion, meridional_localisation)
#(u_pert, v_pert, w_pert) = sphere_to_cartesian(mesh, zonal_pert, meridional_pert)
#perturbation = Function(Vu).project(as_vector([u_pert, v_pert, w_pert]))

# -------------------------------------------------------------- #
# Configuring fields
# -------------------------------------------------------------- #
# get components of u in spherical polar coordinates
zonal_u = u0 * r / a * cos(lat) + zonal_pert
merid_u = Constant(0.0) + meridional_pert
radial_u = Constant(0.0)

# now convert to global Cartesian coordinates
(u_expr, v_expr, w_expr) = sphere_to_cartesian(mesh, zonal_u, merid_u)

# obtain initial conditions
print('Set up initial conditions')
print('project u')
u.project(as_vector([u_expr, v_expr, w_expr]))
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
