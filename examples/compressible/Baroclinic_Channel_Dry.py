from gusto import *
from gusto import thermodynamics
from firedrake import (PeriodicRectangleMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, sin, pi, sqrt,
                       ln, exp, Constant, Function, DirichletBC, as_vector,
                       FunctionSpace, BrokenElement, VectorFunctionSpace,
                       errornorm, norm, cross, grad)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

days = 12 # suggested is 15
dt = 300.0
Lx = 4.0e7  # length
Ly = 6.0e6  # width
H = 3.0e4  # height
degree = 1
omega = Constant(7.292e-5)
phi0 = Constant(pi/4)
tmax = days * 24 * 60 * 60
deltax = 2.5e5
deltay = deltax
deltaz = 1.5e3
dumpfreq = int(tmax / (3 * days * dt))

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
nlayers = int(H/deltaz)
ncolumnsx = int(Lx/deltax)
ncolumnsy = int(Ly/deltay)
m = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly, "x", quadrilateral=True)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, "RTCF", degree)
x,y,z = SpatialCoordinate(mesh)

# Equation
params = CompressibleParameters()
coriolis = 2*omega*sin(phi0)*domain.k
eqns = CompressibleEulerEquations(domain, params, 
                                  Omega=coriolis/2, no_normal_flow_bc_ids=[1, 2])
print(eqns.X.function_space().dim())

# I/O
dirname = 'dry_baroclinic_channel'
output = OutputParameters(dirname=dirname, dump_vtus=False, dumpfreq=dumpfreq, dump_nc=True,
                          dumplist=['cloud_water'])
diagnostic_fields = [Perturbation('theta'), VelocityX, VelocityY, 
                     Temperature(eqns), Pressure(eqns)]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [SSPRK3(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta")]

# Linear solver
linear_solver = CompressibleSolver(eqns)


# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  linear_solver=linear_solver)
                                  
# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

# Physical parameters
a = Constant(6.371229e6)  # radius of earth
b = Constant(2)  # vertical width parameter
beta0 = 2 * omega * cos(phi0) / a
T0 = Constant(288)
Ts = Constant(260)
u0 = Constant(35)
lapse = Constant(0.005)
Rd = params.R_d
Rv = params.R_v
f0 = 2 * omega * sin(phi0)
y0 = Constant(Ly / 2)
g = params.g
p0 = Constant(100000.)
beta0 = Constant(0)
eta_w = Constant(0.3)
deltay_w = Constant(3.2e6)
q0 = Constant(0.016)
cp = params.cp

# Initial conditions
u = stepper.fields("u")
rho = stepper.fields("rho")
theta = stepper.fields("theta")

# spaces
Vu = u.function_space()
Vt = theta.function_space()
Vr = rho.function_space()

# set up background state expressions
eta = Function(Vt).interpolate(Constant(1e-7))
Phi = Function(Vt).interpolate(g * z)
T = Function(Vt)

#Steady State

Phi_prime = u0 / 2 * ((f0 - beta0 * y0) *(y - (Ly/2) - (Ly/(2*pi))*sin(2*pi*y/Ly))
                       + beta0/2*(y**2 - (Ly*y/pi)*sin(2*pi*y/Ly)
                                  - (Ly**2/(2*pi**2))*cos(2*pi*y/Ly) - (Ly**2/3) - (Ly**2/(2*pi**2))))
Phi_expr = (T0 * g / lapse)*(1 - eta**(Rd * lapse / g)) + Phi_prime * ln(eta) * exp(-(ln(eta) / b)**2)
Temp_expr = T0 * eta**(Rd * lapse / g) + (Phi_prime / Rd) * (2/b**2 * ln(eta)**2 - 1) * exp(-(ln(eta) / b)**2)

u_zonal = -u0 ** sin(pi * y / Ly) ** 2 * ln(eta) * eta**(-ln(eta)/b**2)
u_expr = as_vector([u_zonal, 0, 0])
# iterae to make eta
eta_new = Function(Vt)
F = -Phi + Phi_expr
dF = -Rd * Temp_expr / eta
max_iterations = 40
tolerance = 1e-10
for i in range(max_iterations):
    eta_new.interpolate(eta - F/dF)
    if errornorm(eta_new, eta) / norm(eta) < tolerance:
        eta.assign(eta_new)
        break
    eta.assign(eta_new)

# make mean u and theta
u.project(u_expr)
T.interpolate(Temp_expr)
theta.interpolate(thermodynamics.theta(params, Temp_expr, p0 * eta))
Phi_test = Function(Vt).interpolate(Phi_expr)
print("Error in setting up p:", errornorm(Phi_test, Phi) / norm(Phi))

# Calculate hydrostatic fields
compressible_hydrostatic_balance(eqns, theta, rho,  solve_for_rho=True)

# make mean fields
rho_b = Function(Vr).assign(rho)
u_b = stepper.fields("ubar", space=Vu, dump=False).project(u)
theta_b = Function(Vt).assign(theta)

xc = 2.0e6
yc = 2.5e6
Lp = 6.0e5
up = Constant(1.0)
r = sqrt((x - xc) ** 2 + (y - yc) ** 2)
u_pert = Function(Vu).project(as_vector([up * exp(-(r/Lp)**2), 0.0, 0.0]))

# define initial u
u.assign(u_b + u_pert)

# initialise fields
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)