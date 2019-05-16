from gusto import *
from firedrake import PeriodicSquareMesh, PeriodicIntervalMesh, \
    ExtrudedMesh, Constant, SpatialCoordinate, sqrt, cos, \
    conditional, pi, Function, TestFunction, TrialFunction, \
    LinearVariationalProblem, LinearVariationalSolver, dx
import sys

# Choose falling or rising bubble
falling_bubble = True

if '--running-tests' in sys.argv:
    falling_bubble = True
    tmax = 9.
    dt = 3.
    res = [8, 16]
elif falling_bubble:
    dt = 2.
    tmax = 900.
    res = [64, 320]
else:
    dt = 0.25
    tmax = 1000.
    res = [50, 50]

# Set up parameters and mesh
maxk = 4
gauss_deg = 3
DG_deg = 1
dumpfreq = 80
h_rho_pert = True
hamiltonian = True
upwind_rho = False
vorticity = True
reconstruct_q = False

test_2d = True

fieldlist = ['u', 'rho', 'theta']
if vorticity:
    fieldlist.append('q')

parameters = CompressibleParameters()
diagnostics = Diagnostics('rho', "CompressibleEnergy")

upw = '' if upwind_rho else 'no'
ham = '' if hamiltonian else 'non'
vort = '' if not vorticity else '_vorticity'
rec_q = '' if not reconstruct_q or not vorticity else '_recon_q'
dirname = ("EC_3DB{0}{1}_{2}upwindrho_{3}hamiltonian_DG_deg{4}_res{5}_dt{6}"
           "_maxk{7}_gaussdeg{8}".format(vort, rec_q, upw, ham, DG_deg,
                                         res, dt, maxk, gauss_deg))

if falling_bubble:
    H, L = 6400., 32000.
else:
    H, L = 10000., 10000.

nlayers, columns = res[0], res[1]
if not test_2d:
    m = PeriodicSquareMesh(columns, columns, L, quadrilateral=True)
else:
    m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

if hamiltonian:
    hamiltonian = HamiltonianOptions(no_u_rec=(not upwind_rho or vorticity))
timestepping = TimesteppingParameters(dt=dt, maxk=maxk,
                                      reconstruct_q=reconstruct_q)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq,
                          dumplist=['u', 'q'],
                          perturbation_fields=['theta', 'rho'],
                          log_level='INFO')
diagnostic_fields = [CourantNumber(),
                     CompressibleEnergy(),
                     PotentialVorticity()]

state = State(mesh, vertical_degree=DG_deg, horizontal_degree=DG_deg,
              family=("RTCF" if not test_2d else "CG"),
              hamiltonian=hamiltonian,
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              diagnostic_fields=diagnostic_fields,
              fieldlist=fieldlist)

# interpolate initial conditions
u0 = state.fields('u')
rho0 = state.fields('rho')
theta0 = state.fields('theta')

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# Isentropic background state
Tsurf = Constant(300.)

theta_b = Function(Vt).interpolate(Tsurf)
rho_b = Function(Vr)

# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)

# Set up theta perturbation
x = SpatialCoordinate(mesh)
if falling_bubble:
    xc, yc, zc = 0.5*L, 0.5*L, 3000.
    xr, yr, zr = 4000., 4000., 2000.
    if not test_2d:
        r = sqrt(((x[0]-xc)/xr)**2 + ((x[1]-yc)/yr)**2 + ((x[2]-zc)/zr)**2)
    else:
        r = sqrt(((x[0]-xc)/xr)**2 + ((x[1]-zc)/zr)**2)
    Tdash = 7.5
    theta_pert = conditional(r > 1., 0., -Tdash*(1.+cos(pi*r)))
    theta0.interpolate(theta_b + theta_pert)
else:
    xc, yc, zc = 0.5*L, 0.5*L, 2000.
    rc = 2000.
    if not test_2d:
        r = sqrt((x[0] - xc) ** 2 + (x[1] - yc) ** 2 + (x[2] - zc) ** 2)
    else:
        r = sqrt((x[0] - xc) ** 2 + (x[1] - zc) ** 2)
    Tdash = 1.
    theta_pert = Function(Vt).interpolate(conditional(r > rc, 0., Tdash*cos(pi*r/(2.*rc))**2))
    theta0.assign(theta_b * (theta_pert / 300.0 + 1.0))

if h_rho_pert:
    if not falling_bubble:
        # find perturbed rho
        gamma = TestFunction(Vr)
        rho_trial = TrialFunction(Vr)
        lhs = gamma * rho_trial * dx
        rhs = gamma * (rho_b * theta_b / theta0) * dx
        rho_problem = LinearVariationalProblem(lhs, rhs, rho0)
        rho_solver = LinearVariationalSolver(rho_problem)
        rho_solver.solve()
    else:
        compressible_hydrostatic_balance(state, theta0, rho0, solve_for_rho=True)
else:
    rho0.assign(rho_b)

state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

advected_fields = []
if upwind_rho:
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    advected_fields.append(("rho", ThetaMethod(state, rho0, rhoeqn)))
else:
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity",
                               flux_form=True)
    advected_fields.append(("rho", ForwardEuler(state, rho0, rhoeqn)))

# Euler Poincare split only if advection, forcing weights are equal
if upwind_rho != vorticity or not upwind_rho:
    e_p = True
    U_transport = EulerPoincare
else:
    e_p = False
    U_transport = VectorInvariant

if vorticity:
    # initial q solver
    q0 = state.fields('q')
    initial_vorticity(state, rho0, u0, q0)

    # flux formulation has Dp in q-eqn, qp in u-eqn, so order matters
    qeqn = AdvectionEquation(state, q0.function_space(),
                             ibp=IntegrateByParts.NEVER, flux_form=True)
    advected_fields.append(("q", ThetaMethod(state, q0, qeqn, weight='rho')))

    ueqn = U_transport(state, u0.function_space(), vorticity=True)
    advected_fields.append(("u", ForwardEuler(state, u0, ueqn)))
else:
    ueqn = U_transport(state, u0.function_space())
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))

SUPG = False
if SUPG:
    thetaeqn = SUPGAdvection(state, Vt, equation_form="advective")
else:
    thetaeqn = AdvectionEquation(state, Vt, equation_form="advective",
                                 ibp=IntegrateByParts.TWICE)
advected_fields.append(("theta", ThetaMethod(state, theta0, thetaeqn)))

linear_solver = HybridizedCompressibleSolver(state)

# Set up forcing
if hamiltonian:
    compressible_forcing = HamiltonianCompressibleForcing(state, SUPG=SUPG,
                                                          upwind_d=upwind_rho,
                                                          gauss_deg=gauss_deg,
                                                          euler_poincare=e_p,
                                                          vorticity=vorticity)
else:
    compressible_forcing = CompressibleForcing(state, euler_poincare=e_p,
                                               vorticity=vorticity)

# build time stepper
stepper = CrankNicolson(state, advected_fields,
                        linear_solver, compressible_forcing)

stepper.run(t=0, tmax=tmax)
