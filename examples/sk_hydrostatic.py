from gusto import *
from firedrake import as_vector, SpatialCoordinate,\
    PeriodicRectangleMesh, ExtrudedMesh, exp, sin, Function
import numpy as np
import sys

dt = 25.
if '--running-tests' in sys.argv:
    nlayers = 5  # horizontal layers
    columns = 50  # number of columns
    tmax = dt
else:
    nlayers = 10  # horizontal layers
    columns = 150  # number of columns
    tmax = 60000.0

if '--hybridization' in sys.argv:
    hybridization = True
else:
    hybridization = False

L = 6.0e6
m = PeriodicRectangleMesh(columns, 1, L, 1.e4, quadrilateral=True)

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)

dirname = 'sk_hydrostatic'
if hybridization:
    dirname += '_hybridization'

output = OutputParameters(dirname=dirname,
                          dumpfreq=50,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'])

parameters = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber()]

Omega = as_vector((0., 0., 0.5e-4))

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="RTCF",
              Coriolis=Omega,
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

x, y, z = SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)

theta_b = Function(Vt).interpolate(thetab)
rho_b = Function(Vr)

a = 1.0e5
deltaTheta = 1.0e-2
theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
theta0.interpolate(theta_b + theta_pert)
# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)
rho0.assign(rho_b)
u0.project(as_vector([20.0, 0.0, 0.0]))

state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
ueqn = EulerPoincare(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = SUPGAdvection(state, Vt, supg_params={"dg_direction": "horizontal"})
advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

# Set up linear solver
if hybridization:
    linear_solver = HybridizedCompressibleSolver(state)
else:
    linear_solver = CompressibleSolver(state)

# Set up forcing
# [0,0,2*omega] cross [u,v,0] = [-2*omega*v, 2*omega*u, 0]
balanced_pg = as_vector((0., 1.0e-4*20, 0.))
compressible_forcing = CompressibleForcing(state, extra_terms=balanced_pg)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing)

stepper.run(t=0, tmax=tmax)
