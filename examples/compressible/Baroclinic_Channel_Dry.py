from gusto import *
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

# I/O
dirname = 'dry_baroclinic_channel'
output = OutputParameters(dirname=dirname, dumbVTU=False, dumpfreq=dumpfreq, dump_nc=True,
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

# Physics schemes
#physics_schemes = [(SaturationAdjustment(eqns), ForwardEuler(domain))]

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

Phi_expr = (T0 * g / lapse)*(1 - eta**() ) 