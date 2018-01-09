from gusto import *
from gusto import thermodynamics
from firedrake import as_vector, SpatialCoordinate,\
    PeriodicRectangleMesh, ExtrudedMesh, \
    exp, cos, sin, cosh, sinh, tanh, pi, Function, sqrt
import sys

day = 24.*60.*60.
hour = 60.*60.
dt = 30.
if '--running-tests' in sys.argv:
    tmax = dt
    tdump = dt
else:
    tmax = 30*day
    tdump = 2*hour

##############################################################################
# set up mesh
##############################################################################
# Construct 1d periodic base mesh
columns = 30  # number of columns
L = 1000000.
m = PeriodicRectangleMesh(columns, 1, 2.*L, 1.e5, quadrilateral=True)

# build 2D mesh by extruding the base mesh
nlayers = 30  # horizontal layers
H = 10000.  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

##############################################################################
# set up all the other things that state requires
##############################################################################
# Coriolis expression
f = 1.e-04
Omega = as_vector([0., 0., f*0.5])

# list of prognostic fieldnames
# this is passed to state and used to construct a dictionary,
# state.field_dict so that we can access fields by name
# u is the 3D velocity
# p is the pressure
# b is the buoyancy
fieldlist = ['u', 'rho', 'theta']

# class containing timestepping parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
timestepping = TimesteppingParameters(dt=dt)

# class containing output parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
output = OutputParameters(dirname='compressible_eady',
                          dumpfreq=int(tdump/dt),
                          dumplist=['u', 'rho', 'theta'],
                          perturbation_fields=['rho', 'theta', 'ExnerPi'])

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
parameters = CompressibleEadyParameters(H=H, f=f)

# class for diagnostics
# fields passed to this class will have basic diagnostics computed
# (eg min, max, l2 norm) and these will be output as a json file
diagnostics = Diagnostics(*fieldlist)

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber(), VelocityY(),
                     ExnerPi(), ExnerPi(reference=True),
                     CompressibleKineticEnergy(),
                     CompressibleKineticEnergyY(),
                     CompressibleEadyPotentialEnergy(),
                     Sum("CompressibleKineticEnergy",
                         "CompressibleEadyPotentialEnergy"),
                     Difference("CompressibleKineticEnergy",
                                "CompressibleKineticEnergyY")]

# setup state, passing in the mesh, information on the required finite element
# function spaces and the classes above
state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="RTCF",
              Coriolis=Omega,
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

##############################################################################
# Initial conditions
##############################################################################
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# first setup the background buoyancy profile
# z.grad(bref) = N**2
# the following is symbolic algebra, using the default buoyancy frequency
# from the parameters class.
x, y, z = SpatialCoordinate(mesh)
g = parameters.g
Nsq = parameters.Nsq
theta_surf = parameters.theta_surf

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
theta_ref = theta_surf*exp(Nsq*(z-H/2)/g)
theta_b = Function(Vt).interpolate(theta_ref)


# set theta_pert
def coth(x):
    return cosh(x)/sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))


a = -4.5
Bu = 0.5
theta_exp = a*theta_surf/g*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)
                                      - n()*Bu*cosh(Z(z))*sin(pi*(x-L)/L))
theta_pert = Function(Vt).interpolate(theta_exp)

# set theta0
theta0.interpolate(theta_b + theta_pert)

# calculate hydrostatic Pi
rho_b = Function(Vr)
compressible_hydrostatic_balance(state, theta_b, rho_b)
compressible_hydrostatic_balance(state, theta0, rho0)

# set Pi0
Pi0 = calculate_Pi0(state, theta0, rho0)
state.parameters.Pi0 = Pi0

# set x component of velocity
cp = state.parameters.cp
dthetady = state.parameters.dthetady
Pi = thermodynamics.pi(state.parameters, rho0, theta0)
u = cp*dthetady/f*(Pi-Pi0)

# set y component of velocity
v = Function(Vr).assign(0.)
compressible_eady_initial_v(state, theta0, rho0, v)

# set initial u
u_exp = as_vector([u, v, 0.])
u0.project(u_exp)

# pass these initial conditions to the state.initialise method
state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])

# set the background profiles
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

##############################################################################
# Set up advection schemes
##############################################################################
# we need a DG funciton space for the embedded DG advection scheme
ueqn = AdvectionEquation(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = SUPGAdvection(state, Vt, supg_params={"dg_direction": "horizontal"})

advected_fields = []
advected_fields.append(("u", SSPRK3(state, u0, ueqn)))
advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
# Set up linear solver
linear_solver_params = {'ksp_monitor_true_residual': False,
                        'fieldsplit_1_pc_gamg_sym_graph': True,
                        'fieldsplit_1_mg_levels_ksp_max_it': 5}

linear_solver = CompressibleSolver(state, solver_parameters=linear_solver_params)

##############################################################################
# Set up forcing
##############################################################################
forcing = CompressibleEadyForcing(state, euler_poincare=False)

##############################################################################
# build time stepper
##############################################################################
stepper = CrankNicolson(state, advected_fields, linear_solver, forcing)

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax)
