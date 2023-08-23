from firedrake import (ExtrudedMesh, functionspaceimpl,
                       SpatialCoordinate, cos, sin, pi, sqrt, File,
                       exp, Constant, Function, as_vector, acos,
                       errornorm, norm, min_value, max_value, le, ge)
from gusto import *                                            # 
# -------------------------------------------------------------- #
# Test case Parameters
# -------------------------------------------------------------- #
dt = 270.
days = 15.
tmax = days * 24. * 60. * 60.
n = 25     # cells per cubed sphere face edge
deltaz = 2.0e3 # 15 layers, as we are in a higher space this matches the paper better

# --------------------------------------------------------------#
# Script Options
# -------------------------------------------------------------- #
perturbed = False
if perturbed == True:
    dirname = 'baroclinic_wave_'
else: 
    dirname = 'baroclinic_sbr_'

u_form = 'vector_advection_form'
dirname = f'{dirname}{u_form}_'

variable_height = False
limit_theta = False
if limit_theta:
    dirname = f'{dirname}theta_limited_'

# -------------------------------------------------------------- #
# Set up Model
# -------------------------------------------------------------- #
# Domain
a = 6.371229e6  # radius of earth
ztop = 3.0e4  # max height
nlayers = int(ztop/deltaz) 

if variable_height == True: 
    dirname = f'{dirname}variable_height_'
    layerheight=[]
    runningheight=0
    # Calculating Non-uniform height field
    for n in range(1,16):
        mu = 8
        height = ztop * ((mu * (n / 15)**2 + 1)**0.5 - 1) / ((mu + 1)**0.5 - 1)
        width = height - runningheight
        runningheight = height
        layerheight.append(width)
else: 
    layerheight = ztop / nlayers

m = GeneralCubedSphereMesh(a, num_cells_per_edge_of_panel=n, degree=2)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=layerheight, extrusion_type='radial')
lat, lon = latlon_coords(mesh)
domain = Domain(mesh, dt, "RTCF", degree=1)

# Equations
params = CompressibleParameters()
omega = Constant(7.292e-5)
Omega = as_vector((0, 0, omega))
print('making eqn')    
eqn = CompressibleEulerEquations(domain, params, Omega=Omega, u_transport_option=u_form)
print(eqn.X.function_space().dim())

dirname = f'{dirname}dt={dt}_n={n}'
output = OutputParameters(dirname=dirname,
                          dumpfreq=40,
                          dump_nc=True,
                          dump_vtus=False)
diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'),
                     RadialComponent('u'), CourantNumber(), Temperature(eqn), Pressure(eqn), 
                     SteadyStateError('Temperature'), SteadyStateError('Pressure_Vt')]
          
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport Schemes 
if limit_theta:
    Vtheta = domain.spaces("theta")
    limiter = ThetaLimiter(Vtheta)
    options=EmbeddedDGOptions()
else:
    limiter = None 
    options = SUPGOptions()   

transport_option=SUPGOptions()
transported_fields = []
transported_fields.append(TrapeziumRule(domain, "u", options=transport_option))
transported_fields.append(SSPRK3(domain, "rho"))
transported_fields.append(SSPRK3(domain, "theta", options=transport_option))

transport_methods = [DGUpwind(eqn, 'u'),
                     DGUpwind(eqn, 'rho'),
                     DGUpwind(eqn, 'theta', ibp=transport_option.ibp)]


# Linear Solver
linear_solver = CompressibleSolver(eqn)

# Time Stepper
stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                  transport_methods,
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

lapse = 0.005
T0e = 310 # Equatorial temp
T0p = 240 # Polar surface temp
T0 = 0.5 * (T0e + T0p)
H = Rd * T0 / g # scale height of atmosphere
k = 3 # power of temp field
b = 2 # half width parameter

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

tao1_int = A * (exp(lapse * (r - a) / T0) - 1) + B * (r - a) * exp(-((r-a)/(b*H))**2)
tao2_int = C * (r - a)  * exp(-((r-a) / (b*H))**2)

# Variable fields
Temp = (a / r)**2 * (tao1 - tao2 * ( s**k - (k / (k+2)) *s**(k+2)))**(-1)
P_expr = p0 * exp(-g / Rd * tao1_int + g / Rd * tao2_int * (s**k - (k / (k+2)) *s**(k+2)))
wind = ((g*k) / (2 * omega * a)) * (cos(lat)**(k-1) - cos(lat)**(k+1))*tao2_int*Temp

theta_expr = Temp * (P_expr / p0) ** (-params.kappa) 
pie_expr = Temp / theta_expr
rho_expr = P_expr / (Rd * Temp)

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
zeta = conditional(ge(depth,zt-err_tol), 0, 1 - 3*(depth / zt)**2 + 2*(depth / zt)**3) # peturbation vertical taper

perturb_magnitude = (16*Vp/(3*sqrt(3))) * zeta * sin((pi * d) / (2 * d0)) * cos((pi * d) / (2 * d0)) ** 3


zonal_pert = conditional(le(d,err_tol), 0, 
                         conditional(ge(d,(d0-err_tol)), 0, -perturb_magnitude * (-sin(lat_c)*cos(lat) + cos(lat_c)*sin(lat)*cos(lon - lon_c)) / sin(d / a)))
meridional_pert = conditional(le(d,err_tol), 0, 
                              conditional(ge(d,d0-err_tol), 0, perturb_magnitude * cos(lat_c)*sin(lon - lon_c) / sin(d / a)))

rho_expr = P_expr / (Rd * Temp)
# -------------------------------------------------------------- #
# Configuring fields
# -------------------------------------------------------------- #
# get components of u in spherical polar coordinates

zonal_u = wind 
merid_u = Constant(0.0) 
radial_u = Constant(0.0)

#if perturbed == True:
#    zonal_u = zonal_u + zonal_pert
#    merid_u = meridional_pert

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
print('Intialise Windy Boi')
stepper.run(t=0, tmax=tmax)
