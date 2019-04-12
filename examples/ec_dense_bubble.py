from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, Constant, \
    SpatialCoordinate, sqrt, cos, conditional, pi, Function
import sys

dt = 0.5
if '--running-tests' in sys.argv:
    tmax = 5.
    res = [16, 40]
else:
    tmax = 900.
    res = [64, 320]

# Set up parameters and mesh
maxk = 12
gauss_deg = 8
dumpfreq = 36
h_rho_pert = True
upwind_rho = True
fieldlist = ['u', 'rho', 'theta']

H, L = 6400., 32000.
parameters = CompressibleParameters()
diagnostics = Diagnostics('rho', "CompressibleEnergy")

upw = '' if upwind_rho else 'no'
dirname = ("EC_DB_{0}upwindrho_res{1}_dt{2}_maxk{3}_gaussdeg"
           "{4}".format(upw, res, dt, maxk, gauss_deg))

nlayers, columns = res[0], res[1]
m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

timestepping = TimesteppingParameters(dt=dt, maxk=maxk)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'],
                          log_level='INFO')
diagnostic_fields = [CourantNumber(),
                     CompressibleEnergy(),
                     PotentialVorticity()]

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              hamiltonian=True,
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

x = SpatialCoordinate(mesh)
xc = 0.5*L
xr = 4000.
zc = 3000.
zr = 2000.
r = sqrt(((x[0]-xc)/xr)**2 + ((x[1]-zc)/zr)**2)
theta_pert = conditional(r > 1., 0., -7.5*(1.+cos(pi*r)))
theta0.interpolate(theta_b + theta_pert)
if h_rho_pert:
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
    ueqn = EulerPoincare(state, Vu)
    e_p = True
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("rho", ThetaMethod(state, rho0, rhoeqn)))
else:
    ueqn = VectorInvariant(state, Vu)
    e_p = False
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity",
                               flux_form=True)
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("rho", ForwardEuler(state, rho0, rhoeqn)))

thetaeqn = SUPGAdvection(state, Vt, equation_form="advective")
advected_fields.append(("theta", ThetaMethod(state, theta0, thetaeqn)))

linear_solver = HybridizedCompressibleSolver(state)

# Set up forcing
compressible_forcing = HamiltonianCompressibleForcing(state,
                                                      upwind_d=upwind_rho,
                                                      gauss_deg=gauss_deg,
                                                      euler_poincare=e_p)

# build time stepper
stepper = CrankNicolson(state, advected_fields,
                        linear_solver, compressible_forcing)

stepper.run(t=0, tmax=tmax)
