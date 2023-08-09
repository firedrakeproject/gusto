from firedrake import (ExtrudedMesh, SpatialCoordinate, cos, sin, pi, sqrt,
                       exp, Constant, Function, as_vector, acos,
                       errornorm, norm, min_value, max_value, le, ge)
from gusto import *
from gusto.logging import logger
info = logger.info


# ----------------------- #
#  Test case Parameters   #
# ----------------------- #
dt = 270.
days = 10.
tmax = days * 24. * 60. * 60.
deltaz = 2e3  # 15 layers, as we are in a higher space this matches the paper better

# --------------- #
#  Set up Model   #
# --------------- #

# Domain
a = 6.371229e6  # radius of earth
Height = 3.0e4  # height
nlayers = int(Height/deltaz)

m = GeneralCubedSphereMesh(a, num_cells_per_edge_of_panel=10, degree=2)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Height/nlayers, extrusion_type='radial')
domain = Domain(mesh, dt, "RTCF", degree=1)

# Equations
params = CompressibleParameters()
omega = Constant(7.292e-5)
Omega = as_vector((0, 0, omega))

eqn = CompressibleEulerEquations(domain, params, Omega=Omega, u_transport_option='vector_invariant_form')
info(f'Number of degrees of freedom {eqn.X.function_space().dim()}')

dirname = 'baroclinicPerturbation_thetalimiter'
output = OutputParameters(dirname=dirname,
                          dumpfreq=40,
                          dump_nc=True,
                          dump_vtus=False)
diagnostic_fields = [
    MeridionalComponent('u'), ZonalComponent('u'),
    RadialComponent('u'), CourantNumber(), ZonalComponent('u_pert'),
    MeridionalComponent('u_pert'), Temperature(eqn), Pressure(eqn),
    SteadyStateError('Temperature'), SteadyStateError('Pressure_Vt')
]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)
Vtheta = domain.spaces("theta")
limiter = ThetaLimiter(Vtheta)

# Transport Schemes
transported_fields = []
transported_fields.append(TrapeziumRule(domain, "u"))
transported_fields.append(SSPRK3(domain, "rho"))
transported_fields.append(SSPRK3(domain, "theta", options=EmbeddedDGOptions(), limiter=limiter))
transport_methods = [DGUpwind(eqn, field) for field in ["u", "rho", "theta"]]

# Linear Solver
linear_solver = CompressibleSolver(eqn)

# Time Stepper
stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver)

# --------------------- #
#  Initial Conditions   #
# --------------------- #

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

lapse = 0.005
# Equatorial temp
T0e = 310
# Polar surface temp
T0p = 240
T0 = 0.5 * (T0e + T0p)
# scale height of atmosphere
H = Rd * T0 / g
# power of temp field
k = 3
# half width parameter
b = 2

u = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# spaces
Vu = u.function_space()
Vr = rho0.function_space()
Vt = theta0.function_space()

# -------------------------------------------------------------- #
# Base State
# -------------------------------------------------------------- #

# expressions for variables from paper
s = (r / a) * cos(lat)
A = 1 / lapse
B = (T0e - T0p) / ((T0e + T0p)*T0p)
C = ((k + 2) / 2)*((T0e - T0p) / (T0e * T0p))

tao1 = A * lapse / T0 * exp((r - a)*lapse / T0) + B * (1 - 2*((r-a)/(b*H))**2)*exp(-((r-a) / (b*H))**2)
tao2 = C * (1 - 2*((r-a)/(b*H))**2)*exp(-((r - a) / (b*H))**2)

tao1_int = A * (exp(lapse * (r - a) / T0) - 1) + B * (r - a) * exp(-((r-a) / (b*H))**2)
tao2_int = C * (r - a) * exp(-((r-a) / (b*H))**2)

# Variable fields
Temp = (a / r)**2 * (tao1 - tao2 * (s**k - (k / (k+2)) * s**(k+2)))**(-1)
P_expr = p0 * exp(-g / Rd * tao1_int + g / Rd * tao2_int * (s**k - (k / (k+2)) * s**(k+2)))
wind = ((g*k) / (2 * omega * a)) * (cos(lat)**(k-1) - cos(lat)**(k+1))*tao2_int*Temp

theta_expr = Temp * (P_expr / p0) ** (-params.kappa)
pie_expr = Temp / theta_expr
rho_expr = P_expr / (Rd * Temp)


# -------------- #
# Perturbation   #
# -------------- #

# top of perturbation
zt = 1.5e4
# horizontal radius of perturbation
d0 = a / 6
# Perturbed wind amplitude
Vp = 1
# location of perturbation centre
lon_c, lat_c = pi/9, 2*pi/9
err_tol = 1e-12

# distance from centre of perturbation
d = a * acos(sin(lat_c)*sin(lat) + cos(lat_c)*cos(lat)*cos(lon - lon_c))

# The distance from origin subtracted from earth radius
depth = r - a
# peturbation vertical taper
zeta = conditional(
    ge(depth, zt-err_tol), 0,
    1 - 3*(depth / zt)**2 + 2*(depth / zt)**3
)

perturb_magnitude = (16*Vp/(3*sqrt(3))) * zeta * sin((pi * d) / (2 * d0)) * cos((pi * d) / (2 * d0)) ** 3


zonal_pert = conditional(
    le(d, err_tol), 0,
    conditional(
        ge(d, (d0-err_tol)), 0,
        -perturb_magnitude * (-sin(lat_c)*cos(lat) + cos(lat_c)*sin(lat)*cos(lon - lon_c)) / sin(d / a)
    )
)
meridional_pert = conditional(
    le(d, err_tol), 0,
    conditional(
        ge(d, d0-err_tol), 0,
        perturb_magnitude * cos(lat_c)*sin(lon - lon_c) / sin(d / a)
    )
)

rho_expr = P_expr / (Rd * Temp)


# --------------------#
# Configuring fields  #
# --------------------#
# get components of u in spherical polar coordinates
zonal_u = wind + zonal_pert
merid_u = Constant(0.0) + meridional_pert
radial_u = Constant(0.0)

(u_pert, v_pert, w_pert) = sphere_to_cartesian(mesh, zonal_pert, merid_u)
# now convert to global Cartesian coordinates
(u_expr, v_expr, w_expr) = sphere_to_cartesian(mesh, zonal_u, merid_u)

wind_pert = stepper.fields('u_pert', space=Vu)
wind_pert.project(as_vector([u_pert, v_pert, w_pert]))

# obtain initial conditions
info('Set up initial conditions')
info('project u')
u.project(as_vector([u_expr, v_expr, w_expr]))
info('interpolate theta')
theta0.interpolate(theta_expr)
info('find pi')
pie = Function(Vr).interpolate(pie_expr)
info('find rho')
rho0.interpolate(rho_expr)
compressible_hydrostatic_balance(eqn, theta0, rho0, exner_boundary=pie, solve_for_rho=True)

info('make analytic rho')
rho_analytic = Function(Vr).interpolate(rho_expr)
info(f'Normalised rho error is: {errornorm(rho_analytic, rho0) / norm(rho_analytic)}')

# make mean fields
info('make mean fields')
rho_b = Function(Vr).assign(rho0)
theta_b = Function(Vt).assign(theta0)

# assign reference profiles
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])
info('Intialise Windy Boi')
stepper.run(t=0, tmax=tmax)