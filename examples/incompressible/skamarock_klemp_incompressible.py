"""
The gravity wave test case of Skamarock and Klemp (1994), solved using the
incompressible Boussinesq equations.

Buoyancy is transported using SUPG.
"""

from gusto import *
from firedrake import (as_vector, PeriodicIntervalMesh, ExtrudedMesh,
                       sin, SpatialCoordinate, Function, pi)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 6.
L = 3.0e5  # Domain length
H = 1.0e4  # Height position of the model top

if '--running-tests' in sys.argv:
    tmax = dt
    dumpfreq = 1
    columns = 30  # number of columns
    nlayers = 5  # horizontal layers

else:
    tmax = 3600.
    dumpfreq = int(tmax / (2*dt))
    columns = 300  # number of columns
    nlayers = 10  # horizontal layers

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, 'CG', 1)

# Equation
parameters = CompressibleParameters()
eqns = IncompressibleBoussinesqEquations(domain, parameters)

# I/O
output = OutputParameters(dirname='skamarock_klemp_incompressible',
                          dumpfreq=dumpfreq,
                          dumplist=['u'],
                          log_level='INFO')
# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber(), Divergence(), Perturbation('b')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
b_opts = SUPGOptions()
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "b", options=b_opts)]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "b", ibp=b_opts.ibp)]

# Linear solver
linear_solver = IncompressibleSolver(eqns)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields("u")
b0 = stepper.fields("b")
p0 = stepper.fields("p")

# spaces
Vb = b0.function_space()

x, z = SpatialCoordinate(mesh)

# first setup the background buoyancy profile
# z.grad(bref) = N**2
N = parameters.N
bref = z*(N**2)
# interpolate the expression to the function
b_b = Function(Vb).interpolate(bref)

# setup constants
a = 5.0e3
deltab = 1.0e-2
b_pert = deltab*sin(pi*z/H)/(1 + (x - L/2)**2/a**2)
# interpolate the expression to the function
b0.interpolate(b_b + b_pert)

incompressible_hydrostatic_balance(eqns, b_b, p0)

uinit = (as_vector([20.0, 0.0]))
u0.project(uinit)

# set the background buoyancy
stepper.set_reference_profiles([('b', b_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

# Run!
stepper.run(t=0, tmax=tmax)
