"""
The gravity wave test case of Skamarock and Klemp (1994), solved using the
incompressible Boussinesq equations.

Buoyancy is transported using SUPG.
"""

from gusto import *
from firedrake import (as_vector, PeriodicIntervalMesh, ExtrudedMesh,
                       sin, SpatialCoordinate, Function, pi)
import sys

dt = 6.
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

# set up mesh
L = 3.0e5
m = PeriodicIntervalMesh(columns, L)
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

# output parameters
output = OutputParameters(dirname='skamarock_klemp_incompressible',
                          dumpfreq=dumpfreq,
                          dumplist=['u'],
                          perturbation_fields=['b'],
                          log_level='INFO')

# physical parameters
parameters = CompressibleParameters()

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber(), Divergence()]

# setup state
state = State(mesh,
              dt=dt,
              output=output,
              parameters=parameters,
              diagnostic_fields=diagnostic_fields)

eqns = IncompressibleBoussinesqEquations(state, "CG", 1)

# Initial conditions
u0 = state.fields("u")
b0 = state.fields("b")
p0 = state.fields("p")

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

incompressible_hydrostatic_balance(state, b_b, p0)

uinit = (as_vector([20.0, 0.0]))
u0.project(uinit)

# set the background buoyancy
state.set_reference_profiles([('b', b_b)])

# Set up transport schemes
b_opts = SUPGOptions()
transported_fields = [ImplicitMidpoint(state, "u"),
                      SSPRK3(state, "b", options=b_opts)]

# Set up linear solver for the timestepping scheme
linear_solver = IncompressibleSolver(state, eqns)

# build time stepper
stepper = SemiImplicitQuasiNewton(state, eqns, transported_fields,
                                  linear_solver=linear_solver)

# Run!
stepper.run(t=0, tmax=tmax)
