from gusto import *
from firedrake import as_vector, SpatialCoordinate, \
    PeriodicRectangleMesh, ExtrudedMesh, \
    cos, sin, cosh, sinh, tanh, pi, Function, sqrt
import sys

day = 24.*60.*60.
hour = 60.*60.
dt = 100.
if '--running-tests' in sys.argv:
    tmax = dt
    tdump = dt
    # don't use mumps here, testing the overwrite-solver-options functionality
    linear_solver_params = {'ksp_type': 'gmres',
                            'pc_type': 'fieldsplit',
                            'pc_fieldsplit_type': 'additive',
                            'fieldsplit_0_pc_type': 'lu',
                            'fieldsplit_1_pc_type': 'lu',
                            'fieldsplit_0_ksp_type': 'preonly',
                            'fieldsplit_1_ksp_type': 'preonly'}
    overwrite = True
else:
    tmax = 30*day
    tdump = 2*hour
    # use default linear solver parameters (i.e. mumps)
    linear_solver_params = None
    overwrite = False

##############################################################################
# set up mesh
##############################################################################
# parameters
columns = 30
nlayers = 30
H = 10000.
L = 1000000.
f = 1.e-04

# rescaling
beta = 1.0
f = f/beta
L = beta*L

# Construct 2D periodic base mesh
m = PeriodicRectangleMesh(columns, 1, 2.*L, 1.e5, quadrilateral=True)

# build 3D mesh by extruding the base mesh
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

##############################################################################
# set up all the other things that state requires
##############################################################################
# Coriolis expression
Omega = as_vector([0., 0., f*0.5])

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
output = OutputParameters(dirname='incompressible_eady',
                          dumpfreq=int(tdump/dt),
                          dumplist=['u', 'p', 'b'],
                          perturbation_fields=['p', 'b'])

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
parameters = EadyParameters(H=H, L=L, f=f,
                            deltax=2.*L/float(columns),
                            deltaz=H/float(nlayers),
                            fourthorder=True)

# class for diagnostics
# fields passed to this class will have basic diagnostics computed
# (eg min, max, l2 norm) and these will be output as a json file
diagnostics = Diagnostics(*fieldlist)

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber(), VelocityY(),
                     KineticEnergy(), KineticEnergyY(),
                     EadyPotentialEnergy(),
                     Sum("KineticEnergy", "EadyPotentialEnergy"),
                     Difference("KineticEnergy", "KineticEnergyY"),
                     GeostrophicImbalance(), TrueResidualV()]

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


a = -4.5
Bu = 0.5
b_exp = a*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)
                     - n()*Bu*cosh(Z(z))*sin(pi*(x-L)/L))
b_pert = Function(Vb).interpolate(b_exp)

# set total buoyancy
b0.project(b_b + b_pert)

# calculate hydrostatic pressure
p_b = Function(Vp)
incompressible_hydrostatic_balance(state, b_b, p_b)
incompressible_hydrostatic_balance(state, b0, p0)

# set x component of velocity
dbdy = parameters.dbdy
u = -dbdy/f*(z-H/2)

# set y component of velocity
v = Function(Vp).assign(0.)
eady_initial_v(state, p0, v)

# set initial u
u_exp = as_vector([u, v, 0.])
u0.project(u_exp)

# pass these initial conditions to the state.initialise method
state.initialise([('u', u0),
                  ('p', p0),
                  ('b', b0)])

# set the background profiles
state.set_reference_profiles([('p', p_b),
                              ('b', b_b)])

##############################################################################
# Set up advection schemes
##############################################################################
# we need a DG function space for the embedded DG advection scheme
ueqn = AdvectionEquation(state, Vu)
supg = True
if supg:
    beqn = SUPGAdvection(state, Vb,
                         supg_params={"dg_direction": "horizontal"},
                         equation_form="advective")
else:
    beqn = EmbeddedDGAdvection(state, Vb,
                               equation_form="advective")
advected_fields = []
advected_fields.append(("u", SSPRK3(state, u0, ueqn)))
advected_fields.append(("b", SSPRK3(state, b0, beqn)))

##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
linear_solver = IncompressibleSolver(state, 2.*L, solver_parameters=linear_solver_params, overwrite_solver_parameters=overwrite)

##############################################################################
# Set up forcing
##############################################################################
forcing = EadyForcing(state, euler_poincare=False)

##############################################################################
# build time stepper
##############################################################################
stepper = CrankNicolson(state, advected_fields, linear_solver, forcing)

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax)
