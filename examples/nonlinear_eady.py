from gusto import *
from firedrake import as_vector, SpatialCoordinate, \
    PeriodicRectangleMesh, ExtrudedMesh, \
    cos, sin, cosh, sinh, tanh, pi
import sys

dt = 100.
if '--running-tests' in sys.argv:
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
    tmax = 30*24*60*60.
    # use default linear solver parameters (i.e. mumps)
    linear_solver_params = None

##############################################################################
# set up mesh
##############################################################################
# Construct 1d periodic base mesh
columns = 30  # number of columns
L = 1000000.
m = PeriodicRectangleMesh(columns, 1, 2.*L, 1.e4, quadrilateral=True)

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
output = OutputParameters(dirname='nonlinear_eady', dumpfreq=12*36, dumplist=['u','p'])

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
parameters = EadyParameters(H=H)

# class for diagnostics
# fields passed to this class will have basic diagnostics computed
# (eg min, max, l2 norm) and these will be output as a json file
diagnostics = Diagnostics(*fieldlist)

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber()]

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
# set up functions on the spaces constructed by state
u0, p0, b0 = Function(state.V[0]), Function(state.V[1]), Function(state.V[2])

# first setup the background buoyancy profile
# z.grad(bref) = N**2
# the following is symbolic algebra, using the default buoyancy frequency
# from the parameters class.
x, y, z = SpatialCoordinate(mesh)
Nsq = parameters.Nsq
bref = z*Nsq
# interpolate the expression to the function
b_b = Function(state.V[2]).interpolate(bref)


# setup constants
def coth(x):
    return cosh(x)/sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))


a = -7.5
Bu = 0.5
b_exp = a*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)-n()*Bu*cosh(Z(z))*sin(pi*(x-L)/L))
b_pert = Function(state.V[2]).interpolate(b_exp)

# interpolate the expression to the function
b0.interpolate(b_b + b_pert)

# pass these initial conditions to the state.initialise method
state.initialise([u0, p0, b0])
# set the background buoyancy
state.set_reference_profiles({'b':b_b})
# we want to output the perturbation buoyancy, so tell the dump method
# which background field to subtract
state.output.meanfields = ['b']

##############################################################################
# Set up advection schemes
##############################################################################
# we need a DG funciton space for the embedded DG advection scheme
Vtdg = FunctionSpace(mesh, "DG", 1)
# advection_dict is a dictionary containing field_name: advection class
advection_dict = {}
advection_dict["u"] = EulerPoincareForm(state, state.V[0])
advection_dict["b"] = EmbeddedDGAdvection(state, state.V[2], Vdg=Vtdg, continuity=False)

##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
linear_solver = IncompressibleSolver(state, 2.*L, params=linear_solver_params)

##############################################################################
# Set up forcing
##############################################################################
forcing = EadyForcing(state)

##############################################################################
# build time stepper
##############################################################################
stepper = Timestepper(state, advection_dict, linear_solver,
                      forcing)

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax)
