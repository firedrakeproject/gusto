from gusto import *
from firedrake import as_vector,\
    VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh, \
    sin, SpatialCoordinate, Function
import numpy as np
import sys

dt = 6.
if '--running-tests' in sys.argv:
    tmax = dt
    # avoid using mumps on Travis
    linear_solver_params = {'ksp_type': 'gmres',
                            'pc_type': 'fieldsplit',
                            'pc_fieldsplit_type': 'additive',
                            'fieldsplit_0_pc_type': 'lu',
                            'fieldsplit_1_pc_type': 'lu',
                            'fieldsplit_0_ksp_type': 'preonly',
                            'fieldsplit_1_ksp_type': 'preonly'}
else:
    tmax = 3600.
    # use default linear solver parameters (i.e. mumps)
    linear_solver_params = None

##############################################################################
# set up mesh
##############################################################################
# Construct 1d periodic base mesh
columns = 300  # number of columns
L = 3.0e5
m = PeriodicIntervalMesh(columns, L)

# build 2D mesh by extruding the base mesh
nlayers = 10  # horizontal layers
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

##############################################################################
# set up all the other things that state requires
##############################################################################

# list of prognostic fieldnames
# this is passed to state and used to construct a dictionary,
# state.field_dict so that we can access fields by name
# u is the 2D velocity
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
output = OutputParameters(dirname='gw_incompressible', dumpfreq=10, dumplist=['u'], perturbation_fields=['b'])

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
parameters = CompressibleParameters()

# class for diagnostics
# fields passed to this class will have basic diagnostics computed
# (eg min, max, l2 norm) and these will be output as a json file
diagnostics = Diagnostics(*fieldlist)

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber()]

# setup state, passing in the mesh, information on the required finite element
# function spaces, z, k, and the classes above
state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
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
u0 = state.fields("u")
b0 = state.fields("b")
p0 = state.fields("p")

# spaces
Vu = u0.function_space()
Vb = b0.function_space()

x, z = SpatialCoordinate(mesh)

# first setup the background buoyancy profile
# z.grad(bref) = N**2
# the following is symbolic algebra, using the default buoyancy frequency
# from the parameters class.
N = parameters.N
bref = z*(N**2)
# interpolate the expression to the function
b_b = Function(Vb).interpolate(bref)

# setup constants
a = 5.0e3
deltab = 1.0e-2
b_pert = deltab*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
# interpolate the expression to the function
b0.interpolate(b_b + b_pert)

incompressible_hydrostatic_balance(state, b_b, p0)

# interpolate velocity to vector valued function space
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
uinit = Function(W_VectorCG1).interpolate(as_vector([20.0, 0.0]))
# project to the function space we actually want to use
# this step is purely because it is not yet possible to interpolate to the
# vector function spaces we require for the compatible finite element
# methods that we use
u0.project(uinit)

# pass these initial conditions to the state.initialise method
state.initialise([('u', u0),
                  ('b', b0)])
# set the background buoyancy
state.set_reference_profiles([('b', b_b)])

##############################################################################
# Set up advection schemes
##############################################################################
# advected_fields is a dictionary containing field_name: advection class
ueqn = EulerPoincare(state, Vu)
supg = True
if supg:
    beqn = SUPGAdvection(state, Vb,
                         supg_params={"dg_direction": "horizontal"},
                         equation_form="advective")
else:
    beqn = EmbeddedDGAdvection(state, Vb,
                               equation_form="advective")
advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("b", SSPRK3(state, b0, beqn)))

##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
linear_solver = IncompressibleSolver(state, L, params=linear_solver_params)

##############################################################################
# Set up forcing
##############################################################################
forcing = IncompressibleForcing(state)

##############################################################################
# build time stepper
##############################################################################
stepper = Timestepper(state, advected_fields, linear_solver,
                      forcing)

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax)
