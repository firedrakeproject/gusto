from gusto import *
from firedrake import (IntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt,
                       TestFunction, dx, TrialFunction, Constant, Function,
                       LinearVariationalProblem, LinearVariationalSolver,
                       FunctionSpace, BrokenElement, VectorFunctionSpace)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 100.
    tmax = 1000.

if '--recovered' in sys.argv:
    recovered = True
else:
    recovered = False
if '--limit' in sys.argv:
    limit = True
else:
    limit = False


# make mesh
L = 10000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)
m = IntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

# options
diffusion = True
degree = 0 if recovered else 1

dirname = 'dry_bf_bubble'

if recovered:
    dirname += '_recovered'
if limit:
    dirname += '_limit'

output = OutputParameters(dirname=dirname,
                          dumpfreq=20,
                          dumplist=['u'],
                          perturbation_fields=['theta'],
                          log_level='INFO')

params = CompressibleParameters()

state = State(mesh,
              dt=dt,
              output=output,
              parameters=params)

if diffusion:
    diffusion_options = [('u', DiffusionParameters(kappa=60., mu=10./deltax))]
else:
    diffusion_options = None

u_transport_option = "vector_advection_form" if recovered else "vector_invariant_form"

eqns = CompressibleEulerEquations(state, "CG", degree,
                                  u_transport_option=u_transport_option,
                                  diffusion_options=diffusion_options,
                                  no_normal_flow_bc_ids=[1, 2])

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = state.spaces("HDiv")
Vt = state.spaces("theta")
Vr = state.spaces("DG")
x, z = SpatialCoordinate(mesh)

# Define constant theta_e and water_t
Tsurf = 300.0
theta_b = Function(Vt).interpolate(Constant(Tsurf))

# Calculate hydrostatic fields
compressible_hydrostatic_balance(state, theta_b, rho0, solve_for_rho=True)

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
theta0.assign(theta_b * (theta_pert / 300.0 + 1.0))

# find perturbed rho
gamma = TestFunction(Vr)
rho_trial = TrialFunction(Vr)
lhs = gamma * rho_trial * dx
rhs = gamma * (rho_b * theta_b / theta0) * dx
rho_problem = LinearVariationalProblem(lhs, rhs, rho0)
rho_solver = LinearVariationalSolver(rho_problem)
rho_solver.solve()

state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up transport schemes
if recovered:
    VDG1 = state.spaces("DG1", "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
    Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element())
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)
    Vu_brok = FunctionSpace(mesh, BrokenElement(Vu.ufl_element()))

    u_opts = RecoveredOptions(embedding_space=Vu_DG1,
                              recovered_space=Vu_CG1,
                              broken_space=Vu_brok,
                              boundary_method=Boundary_Method.dynamics)
    rho_opts = RecoveredOptions(embedding_space=VDG1,
                                recovered_space=VCG1,
                                broken_space=Vr,
                                boundary_method=Boundary_Method.dynamics)
    theta_opts = RecoveredOptions(embedding_space=VDG1,
                                  recovered_space=VCG1,
                                  broken_space=Vt_brok)
else:
    rho_opts = None
    theta_opts = EmbeddedDGOptions()

# set up limiter
if limit:
    if recovered:
        limiter = VertexBasedLimiter(VDG1)
    else:
        limiter = ThetaLimiter(Vt)
else:
    limiter = None

transported_fields = [SSPRK3(state, "rho", options=rho_opts),
                   SSPRK3(state, "theta", options=theta_opts, limiter=limiter)]
if recovered:
    transported_fields.append(SSPRK3(state, "u", options=u_opts))
else:
    transported_fields.append(ImplicitMidpoint(state, "u"))

# Set up linear solver
linear_solver = CompressibleSolver(state, eqns)

diffusion_schemes = []
if diffusion:
    diffusion_schemes.append(BackwardEuler(state, "u"))

# build time stepper
stepper = CrankNicolson(state, eqns, transported_fields,
                        linear_solver=linear_solver,
                        diffusion_schemes=diffusion_schemes)

stepper.run(t=0, tmax=tmax)
