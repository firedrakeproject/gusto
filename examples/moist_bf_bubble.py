from gusto import *
from gusto import thermodynamics
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt,
                       NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction, dx,
                       TrialFunction, Constant, Function,
                       LinearVariationalProblem, LinearVariationalSolver,
                       DirichletBC,
                       FunctionSpace, BrokenElement, VectorFunctionSpace)
import sys

if '--recovered' in sys.argv:
    recovered = True
else:
    recovered = False
if '--limit' in sys.argv:
    limit = True
else:
    limit = False
if '--diffusion' in sys.argv:
    diffusion = True
else:
    diffusion = False

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 100. if recovered else 200
    tmax = 1000.

L = 10000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
degree = 0 if recovered else 1

dirname = 'moist_bf'

if recovered:
    dirname += '_recovered'
if limit:
    dirname += '_limit'
if diffusion:
    dirname += '_diffusion'

output = OutputParameters(dirname=dirname,
                          dumpfreq=20,
                          dumplist=['u'],
                          perturbation_fields=[],
                          log_level='INFO')

params = CompressibleParameters()
diagnostic_fields = [Theta_e(), InternalEnergy(),
                     Perturbation('InternalEnergy'), PotentialEnergy()]
tracers = [WaterVapour(), CloudWater()]

state = State(mesh,
              dt=dt,
              output=output,
              parameters=params,
              diagnostic_fields=diagnostic_fields)

eqns = CompressibleEulerEquations(state, "CG", degree, active_tracers=tracers)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")
water_v0 = state.fields("vapour_mixing_ratio")
water_c0 = state.fields("cloud_liquid_mixing_ratio")
moisture = ["vapour_mixing_ratio", "cloud_liquid_mixing_ratio"]

# spaces
Vu = state.spaces("HDiv")
Vt = state.spaces("theta")
Vr = state.spaces("DG")
x, z = SpatialCoordinate(mesh)
quadrature_degree = (4, 4)
dxp = dx(degree=(quadrature_degree))

# Define constant theta_e and water_t
Tsurf = 320.0
total_water = 0.02
theta_e = Function(Vt).assign(Tsurf)
water_t = Function(Vt).assign(total_water)

# Calculate hydrostatic fields
saturated_hydrostatic_balance(state, theta_e, water_t)

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)
water_vb = Function(Vt).assign(water_v0)
water_cb = Function(Vt).assign(water_t - water_vb)
pibar = thermodynamics.pi(state.parameters, rho_b, theta_b)
Tb = thermodynamics.T(state.parameters, theta_b, pibar, r_v=water_vb)
Ibar = state.fields("InternalEnergybar", Vt, dump=False)
Ibar.interpolate(thermodynamics.internal_energy(
    state.parameters, rho_b, Tb, r_v=water_vb, r_l=water_cb))

# define perturbation
xc = L / 2
zc = 2000.
rc = 2000.
Tdash = 2.0
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
theta_pert = Function(Vt).interpolate(
    conditional(r > rc,
                0.0,
                Tdash * (cos(pi * r / (2.0 * rc))) ** 2))

# define initial theta
theta0.assign(theta_b * (theta_pert / 300.0 + 1.0))

# find perturbed rho
gamma = TestFunction(Vr)
rho_trial = TrialFunction(Vr)
a = gamma * rho_trial * dxp
L = gamma * (rho_b * theta_b / theta0) * dxp
rho_problem = LinearVariationalProblem(a, L, rho0)
rho_solver = LinearVariationalSolver(rho_problem)
rho_solver.solve()

physics_boundary_method = Boundary_Method.physics if recovered else None

# find perturbed water_v
w_v = Function(Vt)
phi = TestFunction(Vt)
rho_averaged = Function(Vt)
rho_recoverer = Recoverer(
    rho0, rho_averaged,
    VDG=FunctionSpace(mesh, BrokenElement(Vt.ufl_element())),
    boundary_method=physics_boundary_method)
rho_recoverer.project()

pi = thermodynamics.pi(state.parameters, rho_averaged, theta0)
p = thermodynamics.p(state.parameters, pi)
T = thermodynamics.T(state.parameters, theta0, pi, r_v=w_v)
w_sat = thermodynamics.r_sat(state.parameters, T, p)

w_functional = (phi * w_v * dxp - phi * w_sat * dxp)
w_problem = NonlinearVariationalProblem(w_functional, w_v)
w_solver = NonlinearVariationalSolver(w_problem)
w_solver.solve()

water_v0.assign(w_v)
water_c0.assign(water_t - water_v0)

state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b),
                              ('vapour_mixing_ratio', water_vb)])

# set up limiter
if limit:
    if recovered:
        VDG1 = state.spaces("DG1", "DG", 1)
        limiter = VertexBasedLimiter(VDG1)
    else:
        limiter = ThetaLimiter(Vt)
else:
    limiter = None


# Set up transport schemes
if recovered:
    VDG1 = state.spaces("DG1", "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
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
    u_transport = SSPRK3(state, "u", options=u_opts)
else:
    rho_opts = None
    theta_opts = EmbeddedDGOptions()
    u_transport = ImplicitMidpoint(state, "u")

transported_fields = [SSPRK3(state, "rho", options=rho_opts),
                      SSPRK3(state, "theta", options=theta_opts, limiter=limiter),
                      SSPRK3(state, "vapour_mixing_ratio", options=theta_opts, limiter=limiter),
                      SSPRK3(state, "cloud_liquid_mixing_ratio", options=theta_opts, limiter=limiter),
                      u_transport]

# Set up linear solver
linear_solver = CompressibleSolver(state, eqns, moisture=moisture)

# diffusion
bcs = [DirichletBC(Vu, 0.0, "bottom"),
       DirichletBC(Vu, 0.0, "top")]

diffusion_schemes = None

if diffusion:
    diffusion_schemes.append(('u', InteriorPenalty(
        state, Vu, kappa=Constant(60.),
        mu=Constant(10./deltax), bcs=bcs)))

# define condensation
physics_list = [Condensation(state)]

# build time stepper
stepper = CrankNicolson(state, eqns, transported_fields,
                        linear_solver=linear_solver,
                        physics_list=physics_list,
                        diffusion_schemes=diffusion_schemes)

stepper.run(t=0, tmax=tmax)
