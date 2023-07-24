"""
The dry rising bubble test from Bryan & Fritsch (2002).

This uses the lowest-order function spaces, with the recovered methods for
transporting the fields. The test also uses a non-periodic base mesh.
"""

from gusto import *
from firedrake import (IntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt,
                       TestFunction, dx, TrialFunction, Constant, Function,
                       LinearVariationalProblem, LinearVariationalSolver,
                       FunctionSpace, VectorFunctionSpace)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 1.0
L = 10000.
H = 10000.

if '--running-tests' in sys.argv:
    deltax = 1000.
    tmax = 5.
else:
    deltax = 100.
    tmax = 1000.

degree = 0
dirname = 'dry_bryan_fritsch'

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
nlayers = int(H/deltax)
ncolumns = int(L/deltax)
m = IntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, "CG", degree)

# Equation
params = CompressibleParameters()
u_transport_option = "vector_advection_form"
eqns = CompressibleEulerEquations(domain, params,
                                  u_transport_option=u_transport_option,
                                  no_normal_flow_bc_ids=[1, 2])

# I/O
output = OutputParameters(dirname=dirname,
                          dumpfreq=int(tmax / (5*dt)),
                          dumplist=['rho'],
                          dump_vtus=False,
                          dump_nc=True,
                          log_level='INFO')
diagnostic_fields = [Perturbation('theta')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes -- set up options for using recovery wrapper
VDG1 = domain.spaces("DG1_equispaced")
VCG1 = FunctionSpace(mesh, "CG", 1)
Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element())
Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

u_opts = RecoveryOptions(embedding_space=Vu_DG1,
                         recovered_space=Vu_CG1,
                         boundary_method=BoundaryMethod.taylor)
rho_opts = RecoveryOptions(embedding_space=VDG1,
                           recovered_space=VCG1,
                           boundary_method=BoundaryMethod.taylor)
theta_opts = RecoveryOptions(embedding_space=VDG1,
                             recovered_space=VCG1)

transported_fields = [SSPRK3(domain, "rho", options=rho_opts),
                      SSPRK3(domain, "theta", options=theta_opts),
                      SSPRK3(domain, "u", options=u_opts)]

transport_methods = [DGUpwind(eqns, field) for field in ["u", "rho", "theta"]]

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
x, z = SpatialCoordinate(mesh)

# Define constant theta_e and water_t
Tsurf = 300.0
theta_b = Function(Vt).interpolate(Constant(Tsurf))

# Calculate hydrostatic fields
compressible_hydrostatic_balance(eqns, theta_b, rho0, solve_for_rho=True)

# make mean fields
rho_b = Function(Vr).assign(rho0)

# define perturbation
xc = L / 2
zc = 2000.
rc = 2000.
Tdash = 2.0
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
theta_pert = Function(Vt).interpolate(conditional(r > rc,
                                                  0.0,
                                                  Tdash * (cos(pi * r / (2.0 * rc))) ** 2))

# define initial theta
theta0.interpolate(theta_b * (theta_pert / 300.0 + 1.0))

# find perturbed rho
gamma = TestFunction(Vr)
rho_trial = TrialFunction(Vr)
lhs = gamma * rho_trial * dx
rhs = gamma * (rho_b * theta_b / theta0) * dx
rho_problem = LinearVariationalProblem(lhs, rhs, rho0)
rho_solver = LinearVariationalSolver(rho_problem)
rho_solver.solve()

stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
