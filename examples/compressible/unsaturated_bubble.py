"""
A moist thermal in an unsaturated atmosphere. This test is based on that of
Grabowski and Clark (1991), and is described in Bendall et al (2020).

As the thermal rises, water vapour condenses into cloud and forms rain.
Limiters are applied to the transport of the water species.
"""
from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt, exp,
                       TestFunction, dx, TrialFunction, Constant, Function,
                       LinearVariationalProblem, LinearVariationalSolver,
                       FunctionSpace, VectorFunctionSpace, errornorm)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 1.0
if '--running-tests' in sys.argv:
    deltax = 240.
    tmax = 10.
    tdump = tmax
else:
    deltax = 20.
    tmax = 600.
    tdump = 100.

L = 3600.
h = 2400.
nlayers = int(h/deltax)
ncolumns = int(L/deltax)
degree = 0

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=h/nlayers)
domain = Domain(mesh, dt, "CG", degree)

# Equation
params = CompressibleParameters()
tracers = [WaterVapour(), CloudWater(), Rain()]
eqns = CompressibleEulerEquations(domain, params,
                                  active_tracers=tracers)

# I/O
dirname = 'unsaturated_bubble'
output = OutputParameters(dirname=dirname, dumpfreq=tdump, dump_nc=True,
                          dumplist=['cloud_water', 'rain'], log_level='INFO')
diagnostic_fields = [RelativeHumidity(eqns), Perturbation('theta'),
                     Perturbation('water_vapour'), Perturbation('rho')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes -- specify options for using recovery wrapper
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
theta_opts = RecoveryOptions(embedding_space=VDG1, recovered_space=VCG1)
limiter = VertexBasedLimiter(VDG1)

transported_fields = [SSPRK3(domain, "u", options=u_opts),
                      SSPRK3(domain, "rho", options=rho_opts),
                      SSPRK3(domain, "theta", options=theta_opts),
                      SSPRK3(domain, "water_vapour", options=theta_opts, limiter=limiter),
                      SSPRK3(domain, "cloud_water", options=theta_opts, limiter=limiter),
                      SSPRK3(domain, "rain", options=theta_opts, limiter=limiter)]

# Linear solver
linear_solver = CompressibleSolver(eqns)

# Physics schemes
# NB: to use wrapper options with Fallout, need to pass field name to time discretisation
physics_schemes = [(Fallout(eqns, 'rain', domain), SSPRK3(domain, field_name='rain', options=theta_opts, limiter=limiter)),
                   (Coalescence(eqns), ForwardEuler(domain)),
                   (EvaporationOfRain(eqns), ForwardEuler(domain)),
                   (SaturationAdjustment(eqns), ForwardEuler(domain))]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  linear_solver=linear_solver,
                                  physics_schemes=physics_schemes)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")
water_v0 = stepper.fields("water_vapour")
water_c0 = stepper.fields("cloud_water")
rain0 = stepper.fields("rain")

# spaces
Vu = domain.spaces("HDiv")
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")
x, z = SpatialCoordinate(mesh)
quadrature_degree = (4, 4)
dxp = dx(degree=(quadrature_degree))

physics_boundary_method = BoundaryMethod.extruded

# Define constant theta_e and water_t
Tsurf = 283.0
psurf = 85000.
exner_surf = (psurf / eqns.parameters.p_0) ** eqns.parameters.kappa
humidity = 0.2
S = 1.3e-5
theta_surf = thermodynamics.theta(eqns.parameters, Tsurf, psurf)
theta_d = Function(Vt).interpolate(theta_surf * exp(S*z))
H = Function(Vt).assign(humidity)

# Calculate hydrostatic fields
unsaturated_hydrostatic_balance(eqns, stepper.fields, theta_d, H,
                                exner_boundary=Constant(exner_surf))

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
rho_recoverer = Recoverer(rho0, rho_averaged, boundary_method=physics_boundary_method)
rho_h = Function(Vr)
w_h = Function(Vt)
delta = 1.0

R_d = eqns.parameters.R_d
R_v = eqns.parameters.R_v
epsilon = R_d / R_v

# make expressions for determining water_v0
exner = thermodynamics.exner_pressure(eqns.parameters, rho_averaged, theta0)
p = thermodynamics.p(eqns.parameters, exner)
T = thermodynamics.T(eqns.parameters, theta0, exner, water_v0)
r_v_expr = thermodynamics.r_v(eqns.parameters, H, T, p)

# make expressions to evaluate residual
exner_ev = thermodynamics.exner_pressure(eqns.parameters, rho_averaged, theta0)
p_ev = thermodynamics.p(eqns.parameters, exner_ev)
T_ev = thermodynamics.T(eqns.parameters, theta0, exner_ev, water_v0)
RH_ev = thermodynamics.RH(eqns.parameters, water_v0, T_ev, p_ev)
RH = Function(Vt)

# set-up rho problem to keep exner constant
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

    RH.interpolate(RH_ev)
    if errornorm(RH, H) < 1e-10:
        break

    # first solve for r_v
    for j in range(max_inner_solve_count):
        w_h.interpolate(r_v_expr)
        water_v0.assign(water_v0 * (1 - delta) + delta * w_h)

        # compute theta_vd
        theta0.interpolate(theta_d * (1 + water_v0 / epsilon))

        # test quality of solution by re-evaluating expression
        RH.interpolate(RH_ev)
        if errornorm(RH, H) < 1e-10:
            break

    # now solve for rho with theta_vd and w_v guesses
    rho_solver.solve()

    # damp solution
    rho0.assign(rho0 * (1 - delta) + delta * rho_h)

    if i == max_outer_solve_count:
        raise RuntimeError('Hydrostatic balance solve has not converged within %i' % i, 'iterations')

# initialise fields
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b),
                                ('water_vapour', water_vb)])
# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
