from gusto import *
from firedrake import as_vector, SpatialCoordinate, \
    PeriodicRectangleMesh, ExtrudedMesh, \
    cos, sin, cosh, sinh, tanh, pi
import sys

dt = 100.
if '--running-tests' in sys.argv:
    tmax_breed = 0.
    tmax = dt
    # avoid using mumps on Travis
    linear_solver_params = {'ksp_type':'gmres',
                            'pc_type':'fieldsplit',
                            'pc_fieldsplit_type':'additive',
                            'fieldsplit_0_pc_type':'lu',
                            'fieldsplit_1_pc_type':'lu',
                            'fieldsplit_0_ksp_type':'preonly',
                            'fieldsplit_1_ksp_type':'preonly'}
else:
    tmax_breed = 3*24*60*60.
    tmax = 30*24*60*60.
    # use default linear solver parameters (i.e. mumps)
    linear_solver_params = None

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
Omega = as_vector([0.,0.,f*0.5])

# list of prognostic fieldnames
# this is passed to state and used to construct a dictionary,
# state.field_dict so that we can access fields by name
# u is the 3D velocity
# p is the pressure
# b is the buoyancy
fieldlist = ['u', 'p', 'b']

# class containing timestepping parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
timestepping = TimesteppingParameters(dt=dt)

# class containing output parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
output = OutputParameters(dirname='nonlinear_eady', dumpfreq=72,
                          dumplist=['u', 'p'],
                          perturbation_fields=['p', 'b'])

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
parameters = EadyParameters(H=H)

# class for diagnostics
# fields passed to this class will have basic diagnostics computed
# (eg min, max, l2 norm) and these will be output as a json file
diagnostics = Diagnostics(*fieldlist)

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber(), MeridionalVelocity(),
                     KineticEnergy(), KineticEnergyV(),
                     EadyPotentialEnergy(),
                     EadyTotalEnergy()]

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
b0 = state.fields("b")
p0 = state.fields("p")

# spaces
Vu = u0.function_space()
Vb = b0.function_space()
Vp = p0.function_space()

# parameters
x, y, z = SpatialCoordinate(mesh)
Nsq = parameters.Nsq
dbdy = parameters.dbdy

# background buoyancy
bref = (z-H/2)*Nsq
b_b = Function(Vb).project(bref)


# buoyancy perturbation
def coth(x):
    return cosh(x)/sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))


a = -35.
Bu = 0.5
b_exp = a*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)
                     - n()*Bu*cosh(Z(z))*sin(pi*(x-L)/L))
b_pert = Function(Vb).interpolate(b_exp)
b_amp = sqrt(assemble(dot(b_pert, b_pert)*dx))

# set total buoyancy
b0.project(b_b + b_pert)

# calculate hydrostatic pressure
p_b = Function(Vp)
incompressible_hydrostatic_balance(state, b_b, p_b)
incompressible_hydrostatic_balance(state, b0, p0)

# set initial u
u_exp = as_vector([-dbdy/f*(z-H/2), 0.0, 0.0])
u0.project(u_exp)

# pass these initial conditions to the state.initialise method
state.initialise({'u':u0, 'p':p0, 'b':b0})

# set the background profiles
state.set_reference_profiles({'p':p_b, 'b':b_b})

##############################################################################
# Set up advection schemes
##############################################################################
# we need a DG funciton space for the embedded DG advection scheme
ueqn = AdvectionEquation(state, Vu)
supg = False
if supg:
    beqn = SUPGAdvection(state, Vb,
                         supg_params={"dg_direction":"horizontal"},
                         equation_form="advective")
else:
    beqn = EmbeddedDGAdvection(state, Vb,
                               equation_form="advective")
advection_dict = {}
advection_dict["u"] = SSPRK3(state, u0, ueqn)
advection_dict["b"] = SSPRK3(state, b0, beqn)

##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
linear_solver = IncompressibleSolver(state, 2.*L, params=linear_solver_params)

##############################################################################
# Set up forcing
##############################################################################
forcing = EadyForcing(state, euler_poincare=False)

##############################################################################
# build time stepper
##############################################################################
stepper = Timestepper(state, advection_dict, linear_solver, forcing)

##############################################################################
# breeding
##############################################################################
stepper.run(t=0, tmax=tmax_breed, diagnostic_everydump=True)

# re-initialise p and b
pdiff = Function(Vp).interpolate(p0-p_b)
bdiff = Function(Vb).interpolate(b0-b_b)

b_amp_breed = sqrt(assemble(dot(bdiff, bdiff)*dx))
coeff = b_amp/b_amp_breed

b0.assign(b_b+bdiff*coeff)
p0.assign(p_b+pdiff*coeff)

# re-initialise u
u0.project(u_exp)

print "Breeding complete. Restart the model."

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax, diagnostic_everydump=True, breeding=True)
