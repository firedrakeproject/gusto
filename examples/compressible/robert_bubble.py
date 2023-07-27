"""
The dry rising bubble test case of Robert (1993).

Potential temperature is transported using the embedded DG technique.
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       Constant, pi, cos, Function, sqrt, conditional)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 1.
L = 1000.
H = 1000.

if '--running-tests' in sys.argv:
    tmax = dt
    dumpfreq = 1
    nlayers = int(H/50.)
    ncolumns = int(L/50.)
else:
    tmax = 600.
    dumpfreq = int(tmax / (6*dt))
    nlayers = int(H/10.)
    ncolumns = int(L/10.)

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, "CG", 1)

# Equation
parameters = CompressibleParameters()
eqns = CompressibleEulerEquations(domain, parameters)

# I/O
dirname = 'robert_bubble'
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist=['u'],
                          log_level='INFO')
diagnostic_fields = [CourantNumber(), Perturbation('theta'), Perturbation('rho')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
theta_opts = EmbeddedDGOptions()
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta", options=theta_opts)]

transport_methods = [DGUpwind(eqns, "u"),
                     DGUpwind(eqns, "rho"),
                     DGUpwind(eqns, "theta")]

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

# Isentropic background state
Tsurf = Constant(300.)

theta_b = Function(Vt).interpolate(Tsurf)
rho_b = Function(Vr)

# Calculate hydrostatic exner
compressible_hydrostatic_balance(eqns, theta_b, rho_b, solve_for_rho=True)

x = SpatialCoordinate(mesh)
xc = 500.
zc = 350.
rc = 250.
r = sqrt((x[0]-xc)**2 + (x[1]-zc)**2)
theta_pert = conditional(r > rc, 0., 0.25*(1. + cos((pi/rc)*r)))

theta0.interpolate(theta_b + theta_pert)
rho0.interpolate(rho_b)

stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
