from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, as_vector, sqrt, sin, exp, pi, \
    FunctionSpace, Constant, Function
import sys

# set up sk nonlinear parameters and mesh
res = [20, 150]
dt = 3.
tmax = 3600.
maxk = 12
gauss_deg = 8
dumpfreq = 10
h_rho_pert=True

# set up input that doesn't change with ref level or dt
fieldlist = ['u', 'rho', 'theta']

H, L = 10000., 300000.
parameters = CompressibleParameters()
diagnostics = Diagnostics('rho', "CompressibleEnergy")

dirname = "EC_SK_res{0}_dt{1}_maxk{2}_gaussdeg{3}".format(res, dt, maxk, gauss_deg)

nlayers, columns = res[0], res[1]
m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

timestepping = TimesteppingParameters(dt=dt, alpha=1., maxk=maxk)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'])
diagnostic_fields = [CourantNumber(),
                     CompressibleEnergy()]

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

# Isothermal background state
x = SpatialCoordinate(mesh)
g, N = parameters.g, parameters.N
Tsurf = 300.
theta_b = Function(Vt).interpolate(Tsurf*exp(N**2*x[1]/g))
rho_b = Function(Vr)

# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)

a, delta_th = 5000., 1.0e-2
theta_pert = delta_th*sin(pi*x[1]/H)/(1 + (x[0] - L/2.)**2/a**2)
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

ueqn = EulerPoincare(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = SUPGAdvection(state, Vt,
                         supg_params={"dg_direction": "horizontal"},
                         equation_form="advective")
advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("rho", ThetaMethod(state, rho0, rhoeqn)))
advected_fields.append(("theta", ThetaMethod(state, theta0, thetaeqn)))

linear_solver = HybridizedCompressibleSolver(state)

# Set up forcing
compressible_forcing = HamiltonianCompressibleForcing(state, gauss_deg=gauss_deg)

# build time stepper
stepper = CrankNicolson(state, advected_fields,
                        linear_solver, compressible_forcing)

stepper.run(t=0, tmax=tmax)
