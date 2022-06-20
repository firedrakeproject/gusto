"""
This test is similar to the one done by Grabowski and Clark (1991),
featuring a moist thermal rising in an unsaturated atmosphere.
"""
from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt, exp,
                       TestFunction, dx, TrialFunction, Constant, Function,
                       LinearVariationalProblem, LinearVariationalSolver,
                       FunctionSpace, BrokenElement, VectorFunctionSpace, errornorm)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import sys

if '--recovered' in sys.argv:
    recovered = True
else:
    recovered = False

if '--diffusion' in sys.argv:
    diffusion = True
else:
    diffusion = False

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 240.
else:
    deltax = 20. if recovered else 40.
    tmax = 600.

L = 3600.
h = 2400.
nlayers = int(h/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=h/nlayers)
degree = 0 if recovered else 1

dirname = 'unsaturated_bubble'
if recovered:
    dirname += '_recovered'
if diffusion:
    dirname += '_diffusion'

output = OutputParameters(dirname=dirname, dumpfreq=20,
                          perturbation_fields=['theta', 'vapour_mixing_ratio', 'rho'],
                          log_level='INFO')
params = CompressibleParameters()
diagnostic_fields = [RelativeHumidity(), Theta_e()]
tracers = [WaterVapour(), CloudWater(), Rain()]

state = State(mesh,
              dt=dt,
              output=output,
              parameters=params,
              diagnostic_fields=diagnostic_fields)

if diffusion:
    diffusion_options = [('u', DiffusionParameters(kappa=60., mu=10./deltax))]
else:
    diffusion_options = None

eqns = CompressibleEulerEquations(state, "CG", degree,
                                  diffusion_options=diffusion_options,
                                  active_tracers=tracers)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")
water_v0 = state.fields("vapour_mixing_ratio")
water_c0 = state.fields("cloud_liquid_mixing_ratio")
rain0 = state.fields("rain", theta0.function_space())
moisture = ["vapour_mixing_ratio", "cloud_liquid_mixing_ratio", "rain_mixing_ratio"]

# spaces
Vu = state.spaces("HDiv")
Vt = state.spaces("theta")
Vr = state.spaces("DG")
Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
x, z = SpatialCoordinate(mesh)
quadrature_degree = (4, 4)
dxp = dx(degree=(quadrature_degree))
physics_boundary_method = None

if recovered:
    VDG1 = state.spaces("DG1", "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element())
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    u_opts = RecoveredOptions(embedding_space=Vu_DG1,
                              recovered_space=Vu_CG1,
                              broken_space=Vu,
                              boundary_method=Boundary_Method.dynamics)
    rho_opts = RecoveredOptions(embedding_space=VDG1,
                                recovered_space=VCG1,
                                broken_space=Vr,
                                boundary_method=Boundary_Method.dynamics)
    theta_opts = RecoveredOptions(embedding_space=VDG1,
                                  recovered_space=VCG1,
                                  broken_space=Vt_brok)
    physics_boundary_method = Boundary_Method.physics

# Define constant theta_e and water_t
Tsurf = 283.0
psurf = 85000.
pi_surf = (psurf / state.parameters.p_0) ** state.parameters.kappa
humidity = 0.2
S = 1.3e-5
theta_surf = thermodynamics.theta(state.parameters, Tsurf, psurf)
theta_d = Function(Vt).interpolate(theta_surf * exp(S*z))
H = Function(Vt).assign(humidity)

# Calculate hydrostatic fields
unsaturated_hydrostatic_balance(state, theta_d, H,
                                pi_boundary=Constant(pi_surf))

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)
water_vb = Function(Vt).assign(water_v0)

# define perturbation to RH
xc = L / 2
zc = 800.
r1 = 300.
r2 = 200.
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)

H_expr = conditional(
    r > r1, 0.0,
    conditional(r > r2,
                (1 - humidity) * cos(pi * (r - r2)
                                     / (2 * (r1 - r2))) ** 2,
                1 - humidity))
H_pert = Function(Vt).interpolate(H_expr)
H.assign(H + H_pert)

# now need to find perturbed rho, theta_vd and r_v
# follow approach used in unsaturated hydrostatic setup
rho_averaged = Function(Vt)
rho_recoverer = Recoverer(rho0, rho_averaged, VDG=Vt_brok,
                          boundary_method=physics_boundary_method)
rho_h = Function(Vr)
w_h = Function(Vt)
delta = 1.0

R_d = state.parameters.R_d
R_v = state.parameters.R_v
epsilon = R_d / R_v

# make expressions for determining water_v0
pie = thermodynamics.pi(state.parameters, rho_averaged, theta0)
p = thermodynamics.p(state.parameters, pie)
T = thermodynamics.T(state.parameters, theta0, pie, water_v0)
r_v_expr = thermodynamics.r_v(state.parameters, H, T, p)

# make expressions to evaluate residual
pi_ev = thermodynamics.pi(state.parameters, rho_averaged, theta0)
p_ev = thermodynamics.p(state.parameters, pi_ev)
T_ev = thermodynamics.T(state.parameters, theta0, pi_ev, water_v0)
RH_ev = thermodynamics.RH(state.parameters, water_v0, T_ev, p_ev)
RH = Function(Vt)

# set-up rho problem to keep Pi constant
gamma = TestFunction(Vr)
rho_trial = TrialFunction(Vr)
a = gamma * rho_trial * dxp
L = gamma * (rho_b * theta_b / theta0) * dxp
rho_problem = LinearVariationalProblem(a, L, rho_h)
rho_solver = LinearVariationalSolver(rho_problem)

max_outer_solve_count = 20
max_inner_solve_count = 10

for i in range(max_outer_solve_count):
    # calculate averaged rho
    rho_recoverer.project()

    RH.assign(RH_ev)
    if errornorm(RH, H) < 1e-10:
        break

    # first solve for r_v
    for j in range(max_inner_solve_count):
        w_h.interpolate(r_v_expr)
        water_v0.assign(water_v0 * (1 - delta) + delta * w_h)

        # compute theta_vd
        theta0.assign(theta_d * (1 + water_v0 / epsilon))

        # test quality of solution by re-evaluating expression
        RH.assign(RH_ev)
        if errornorm(RH, H) < 1e-10:
            break

    # now solve for rho with theta_vd and w_v guesses
    rho_solver.solve()

    # damp solution
    rho0.assign(rho0 * (1 - delta) + delta * rho_h)

    if i == max_outer_solve_count:
        raise RuntimeError('Hydrostatic balance solve has not converged within %i' % i, 'iterations')

# initialise fields
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b),
                              ('vapour_mixing_ratio', water_vb)])

# Set up advection schemes
if recovered:
    u_advection = SSPRK3(state, "u", options=u_opts)
    rho_opts = EmbeddedDGOptions()
    theta_opts = EmbeddedDGOptions()
    limiter = VertexBasedLimiter(VDG1)
else:
    u_advection = ImplicitMidpoint(state, "u")
    rho_opts = None
    theta_opts = EmbeddedDGOptions()

    limiter = ThetaLimiter(Vt)

advected_fields = [u_advection,
                   SSPRK3(state, "rho", options=rho_opts),
                   SSPRK3(state, "theta", options=theta_opts),
                   SSPRK3(state, "vapour_mixing_ratio", options=theta_opts, limiter=limiter),
                   SSPRK3(state, "cloud_liquid_mixing_ratio", options=theta_opts, limiter=limiter),
                   SSPRK3(state, "rain_mixing_ratio", options=theta_opts, limiter=limiter)]

# Set up linear solver
linear_solver = CompressibleSolver(state, eqns, moisture=moisture)

diffusion_schemes = []

if diffusion:
    diffusion_schemes.append(BackwardEuler(state, "u"))

# define condensation
physics_list = [Fallout(state), Coalescence(state), Evaporation(state),
                Condensation(state)]

# build time stepper
stepper = CrankNicolson(state, eqns, advected_fields,
                        linear_solver=linear_solver,
                        physics_list=physics_list,
                        diffusion_schemes=diffusion_schemes)

stepper.run(t=0, tmax=tmax)
