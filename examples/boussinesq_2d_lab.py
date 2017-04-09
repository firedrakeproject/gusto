from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    sin, exp, pi, SpatialCoordinate
import numpy as np
import sympy as sp
from sympy.stats import Normal
import sys

dt = 1./20

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
    tmax = 3600*6.
    # use default linear solver parameters (i.e. mumps)
    linear_solver_params = None

##############################################################################
# set up mesh
##############################################################################
# Construct 1d periodic base mesh for idealised lab experiment of Park et al. (1994)
columns = 20  # number of columns
L = 0.2
m = PeriodicIntervalMesh(columns, L)

# build 2D mesh by extruding the base mesh
nlayers = 45  # horizontal layers
H = 0.45  # Height position of the model top
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
output = OutputParameters(dirname='boussinesq_2d_lab', dumpfreq=200, dumplist=['u','b'], perturbation_fields=['b'])

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
rho_0 = Constant(1090.95075)
#N=1.957 (run 18), N=1.1576 (run 16), N=0.5916 (run 14), N=0.2
parameters = CompressibleParameters(N=1.957, p_0=106141.3045)

# Physical parameters adjusted for idealised lab experiment of Park et al. (1994):
# The value of the background buoyancy frequency N is that for their run number 18, which has clear stair-step features.
# p_0 was set by assuming an initially hydrostatic fluid and a reference density rho_0=1090.95075 kg m^(-3).
# The reference density was found by estimating drho/dz from Fig. 7a of Park et al. (1994), converting to SI units,
# and then using the N^2 value above.
# p_0=106141.3045

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
p0 = state.fields("p")
b0 = state.fields("b")

# first setup the background buoyancy profile
# z.grad(bref) = N**2
# the following is symbolic algebra, using the default buoyancy frequency
# from the parameters class. x[1]=z and comes from x=SpatialCoordinate(mesh)
x = SpatialCoordinate(mesh)
N = parameters.N
bref = N**2*(x[1]-H)
# interpolate the expression to the function
Vb = b0.function_space()
b_b = Function(Vb).interpolate(bref)

# Define constants for bouyancy perturbation:
g = parameters.g
A_x1 = Constant(0) 				# Initial amplitude of internal waves in x-direction (e.g., try setting to zero initially)
A_z1 = Constant(g/rho_0 * 100./3)	 	# Initial amplitude of internal waves in z-direction
lmda_x1 = Constant(1.3/100)			# Horizontal wavelength of internal waves
#lmda_z1 = Constant(1.3/100)			# Vertical wavelength of internal waves
lmda_z1 = Constant(20./100)			# Vertical wavelength of internal waves
k1 = Constant(2*np.pi/lmda_x1)			# Horizontal wavenumber of internal waves
m1 = Constant(2*np.pi/lmda_z1)			# Vertical wavenumber of internal waves

# Define bouyancy perturbation to represent background soup of internal waves in idealised lab scenario of Park et al.
#b_pert = A_x1*sin(k1*x[0]) + A_z1/2.*sin(m1*x[1])
#sigma = 0.01
#b_pert = A_z1*exp( -( x[1] - H/2 )**2 / (2*sigma**2) )

r = Function(b0.function_space()).assign(Constant(0.0))
r.dat.data[:] += np.random.uniform(low=-1., high=1., size=r.dof_dset.size)
b_pert = r*A_z1/2.

#b_pert = sp.Piecewise( (0, x[1] < H/2-0.01), (0, x[1] > H/2+0.01), (A_z1, H/2-0.01 >= x[1] <= H/2+0.01, True) ) - doesn't work
#b_pert = sp.integrate( A_z1 * DiracDelta(x[1]-H/2), (x[1],0,H) ) - doesn't work

# interpolate the expression to the function
b0.interpolate(b_b + b_pert)


incompressible_hydrostatic_balance(state, b_b, p0, top=False)

# pass these initial conditions to the state.initialise method
state.initialise({"u": u0, "p": p0, "b": b0})
# set the background buoyancy
state.set_reference_profiles({"b": b_b})

##############################################################################
# Set up advection schemes
##############################################################################
# advection_dict is a dictionary containing field_name: advection class
ueqn = EulerPoincare(state, u0.function_space())
supg = True
if supg:
    beqn = SUPGAdvection(state, Vb,
                         supg_params={"dg_direction":"horizontal"},
                         equation_form="advective")
else:
    beqn = EmbeddedDGAdvection(state, Vb,
                               equation_form="advective")
advection_dict = {}
advection_dict["u"] = ThetaMethod(state, u0, ueqn)
advection_dict["b"] = SSPRK3(state, b0, beqn)

##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
linear_solver = IncompressibleSolver(state, L, params=linear_solver_params)

##############################################################################
# Set up forcing
##############################################################################
forcing = IncompressibleForcing(state)


##############################################################################
#Set up diffusion scheme
##############################################################################
# mu is a numerical parameter
# kappa is the diffusion constant
#Vu = u0.function_space()
#Vb = b0.function_space()
#delta = ?

#bcs = [DirichletBC(Vu, 0.0, "bottom"), DirichletBC(Vu, 0.0, "top")]

#diffusion_dict = {"u": InteriorPenalty(state, Vu, kappa=Constant(75.),
#                                           mu=Constant(10./delta), bcs=bcs),
#                      "b": InteriorPenalty(state, Vb, kappa=Constant(75.),
#                                               mu=Constant(10./delta))}



##############################################################################
# build time stepper
##############################################################################
stepper = Timestepper(state, advection_dict, linear_solver,
                      forcing)

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax)
