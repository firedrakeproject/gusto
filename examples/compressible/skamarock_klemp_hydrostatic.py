"""
The non-linear gravity wave test case of Skamarock and Klemp (1994), but solved
with the hydrostatic Compressible Euler equations.

Potential temperature is transported using SUPG.
"""

from gusto import *
from firedrake import (as_vector, SpatialCoordinate, PeriodicRectangleMesh,
                       ExtrudedMesh, exp, sin, Function, pi)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 25.
if '--running-tests' in sys.argv:
    nlayers = 5  # horizontal layers
    columns = 10  # number of columns
    tmax = dt
    dumpfreq = 1
else:
    nlayers = 10  # horizontal layers
    columns = 150  # number of columns
    tmax = 60000.0
    dumpfreq = int(tmax / (2*dt))

L = 6.0e6  # Length of domain
H = 1.0e4  # Height position of the model top

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain -- 3D volume mesh
m = PeriodicRectangleMesh(columns, 1, L, 1.e4, quadrilateral=True)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, "RTCF", 1)

# Equation
parameters = CompressibleParameters()
Omega = as_vector((0., 0., 0.5e-4))
balanced_pg = as_vector((0., -1.0e-4*20, 0.))
eqns = CompressibleEulerEquations(domain, parameters, Omega=Omega,
                                  extra_terms=[("u", balanced_pg)])

# I/O
dirname = 'skamarock_klemp_hydrostatic'
output = OutputParameters(
    dirname=dirname,
    dumpfreq=dumpfreq,
    dumplist=['u'],
)
diagnostic_fields = [CourantNumber(), Perturbation('theta'), Perturbation('rho')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
theta_opts = SUPGOptions()
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta", options=theta_opts)]

transport_methods = [DGUpwind(eqns, "u"),
                     DGUpwind(eqns, "rho"),
                     DGUpwind(eqns, "theta", ibp=theta_opts.ibp)]

# Linear solver
linear_solver = CompressibleSolver(eqns)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# spaces
Vu = domain.spaces("HDiv")
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

x, y, z = SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)

theta_b = Function(Vt).interpolate(thetab)
rho_b = Function(Vr)

a = 1.0e5
deltaTheta = 1.0e-2
theta_pert = deltaTheta*sin(pi*z/H)/(1 + (x - L/2)**2/a**2)
theta0.interpolate(theta_b + theta_pert)

compressible_hydrostatic_balance(eqns, theta_b, rho_b,
                                 solve_for_rho=True)

rho0.assign(rho_b)
u0.project(as_vector([20.0, 0.0, 0.0]))

stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
