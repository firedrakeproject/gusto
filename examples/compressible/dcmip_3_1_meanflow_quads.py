"""
The non-orographic gravity wave test case (3-1) from the DCMIP test case
document of Ullrich et al (2012).

This uses a cubed-sphere mesh.
"""

from gusto import *
from firedrake import (CubedSphereMesh, ExtrudedMesh, FunctionSpace,
                       Function, SpatialCoordinate, as_vector)
from firedrake import exp, acos, cos, sin, pi, sqrt, asin, atan_2
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

nlayers = 10           # Number of vertical layers
refinements = 3        # Number of horiz. refinements

dt = 100.0             # Time-step size (s)

if '--running-tests' in sys.argv:
    tmax = dt
    dumpfreq = 1
else:
    tmax = 3600.0
    dumpfreq = int(tmax / (4*dt))

parameters = CompressibleParameters()
a_ref = 6.37122e6               # Radius of the Earth (m)
X = 125.0                       # Reduced-size Earth reduction factor
a = a_ref/X                     # Scaled radius of planet (m)
g = parameters.g                # Acceleration due to gravity (m/s^2)
N = parameters.N                # Brunt-Vaisala frequency (1/s)
p_0 = parameters.p_0            # Reference pressure (Pa, not hPa)
c_p = parameters.cp             # SHC of dry air at constant pressure (J/kg/K)
R_d = parameters.R_d            # Gas constant for dry air (J/kg/K)
kappa = parameters.kappa        # R_d/c_p
T_eq = 300.0                    # Isothermal atmospheric temperature (K)
p_eq = 1000.0 * 100.0           # Reference surface pressure at the equator
u_max = 20.0                    # Maximum amplitude of the zonal wind (m/s)
d = 5000.0                      # Width parameter for Theta'
lamda_c = 2.0*pi/3.0            # Longitudinal centerpoint of Theta'
phi_c = 0.0                     # Latitudinal centerpoint of Theta' (equator)
deltaTheta = 1.0                # Maximum amplitude of Theta' (K)
L_z = 20000.0                   # Vertical wave length of the Theta' perturb.
z_top = 1.0e4                   # Height position of the model top

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
# Cubed-sphere horizontal mesh
m = CubedSphereMesh(radius=a,
                    refinement_level=refinements,
                    degree=2)
# Build volume mesh
mesh = ExtrudedMesh(m, layers=nlayers,
                    layer_height=z_top/nlayers,
                    extrusion_type="radial")
domain = Domain(mesh, dt, "RTCF", 1)
x = SpatialCoordinate(mesh)

# Create polar coordinates:
# Since we use a CG1 field, this is constant on layers
W_Q1 = FunctionSpace(mesh, "CG", 1)
z_expr = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a
z = Function(W_Q1).interpolate(z_expr)
lat_expr = asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))
lat = Function(W_Q1).interpolate(lat_expr)
lon = Function(W_Q1).interpolate(atan_2(x[1], x[0]))

# Equation
eqns = CompressibleEulerEquations(domain, parameters)

# I/O
dirname = 'dcmip_3_1_meanflow'
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          log_level='INFO')
diagnostic_fields = [Perturbation('theta'), Perturbation('rho'), CompressibleKineticEnergy()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [ImplicitMidpoint(domain, "u"),
                      SSPRK3(domain, "rho", subcycles=2),
                      SSPRK3(domain, "theta", options=SUPGOptions(), subcycles=2)]
transport_methods = [DGUpwind(eqns, field) for field in ["u", "rho", "theta"]]

# Linear solver
linear_solver = CompressibleSolver(eqns)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods,
                                  linear_solver=linear_solver)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields('u')
theta0 = stepper.fields('theta')
rho0 = stepper.fields('rho')

# spaces
Vu = domain.spaces("HDiv")
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")

# Initial conditions with u0
uexpr = as_vector([-u_max*x[1]/a, u_max*x[0]/a, 0.0])
u0.project(uexpr)

# Surface temperature
G = g**2/(N**2*c_p)
Ts_expr = G + (T_eq-G)*exp(-(u_max*N**2/(4*g*g))*u_max*(cos(2.0*lat)-1.0))
Ts = Function(W_Q1).interpolate(Ts_expr)

# Surface pressure
ps_expr = p_eq*exp((u_max/(4.0*G*R_d))*u_max*(cos(2.0*lat)-1.0))*(Ts/T_eq)**(1.0/kappa)
ps = Function(W_Q1).interpolate(ps_expr)

# Background pressure
p_expr = ps*(1 + G/Ts*(exp(-N**2*z/g)-1))**(1.0/kappa)
p = Function(W_Q1).interpolate(p_expr)

# Background temperature
Tb_expr = G*(1 - exp(N**2*z/g)) + Ts*exp(N**2*z/g)
Tb = Function(W_Q1).interpolate(Tb_expr)

# Background potential temperature
thetab_expr = Tb*(p_0/p)**kappa
thetab = Function(W_Q1).interpolate(thetab_expr)
theta_b = Function(theta0.function_space()).interpolate(thetab)
rho_b = Function(rho0.function_space())
sin_tmp = sin(lat) * sin(phi_c)
cos_tmp = cos(lat) * cos(phi_c)
r = a*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))
s = (d**2)/(d**2 + r**2)
theta_pert = deltaTheta*s*sin(2*pi*z/L_z)
theta0.interpolate(theta_b)

# Compute the balanced density
compressible_hydrostatic_balance(eqns,
                                 theta_b,
                                 rho_b,
                                 top=False,
                                 exner_boundary=(p/p_0)**kappa)
theta0.interpolate(theta_pert)
theta0 += theta_b
rho0.assign(rho_b)

stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

# Run!
stepper.run(t=0, tmax=tmax)
