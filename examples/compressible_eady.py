from gusto import *
from firedrake import as_vector, SpatialCoordinate,\
    PeriodicRectangleMesh, ExtrudedMesh, exp, sin
import numpy as np
import sys

dt = 30.
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
#    tmax = 30*24*60*60.
    tmax = 30*60.
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
fieldlist = ['u', 'rho', 'theta']

# class containing timestepping parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
timestepping = TimesteppingParameters(dt=dt)

# class containing output parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
output = OutputParameters(dirname='compressible_eady', dumpfreq=1, dumplist=['u','rho','theta'], 
                          perturbation_fields=['rho', 'theta'])

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
parameters = CompressibleEadyParameters(H=H)

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
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa
Nsq = parameters.Nsq

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetaref = Tsurf*exp(Nsq*(z-H/2)/g)
theta_b = Function(Vt).interpolate(thetaref)
rho_b = Function(Vr)

# hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b)
rho0.assign(rho_b)
theta0.interpolate(theta_b)

#theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
#theta0.interpolate(theta_b + theta_pert)
#u0.project(as_vector([20.0, 0.0, 0.0]))

#state.initialise({'u': u0, 'rho': rho0, 'theta': theta0})
state.initialise({'rho': rho0, 'theta': theta0})
state.set_reference_profiles({'rho':rho_b, 'theta':theta_b})

##############################################################################
# Set up advection schemes
##############################################################################
# we need a DG funciton space for the embedded DG advection scheme
ueqn = AdvectionEquation(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = SUPGAdvection(state, Vt, supg_params={"dg_direction":"horizontal"})

advection_dict = {}
advection_dict["u"] = ThetaMethod(state, u0, ueqn)
advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn)
advection_dict["theta"] = SSPRK3(state, theta0, thetaeqn)

##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
# Set up linear solver
schur_params = {'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'ksp_type': 'gmres',
                'ksp_monitor_true_residual': True,
                'ksp_max_it': 100,
                'ksp_gmres_restart': 50,
                'pc_fieldsplit_schur_fact_type': 'FULL',
                'pc_fieldsplit_schur_precondition': 'selfp',
                'fieldsplit_0_ksp_type': 'richardson',
                'fieldsplit_0_ksp_max_it': 5,
                'fieldsplit_0_pc_type': 'bjacobi',
                'fieldsplit_0_sub_pc_type': 'ilu',
                'fieldsplit_1_ksp_type': 'richardson',
                'fieldsplit_1_ksp_max_it': 5,
                "fieldsplit_1_ksp_monitor_true_residual": True,
                'fieldsplit_1_pc_type': 'gamg',
                'fieldsplit_1_pc_gamg_sym_graph': True,
                'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                'fieldsplit_1_mg_levels_ksp_max_it': 5,
                'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

linear_solver = CompressibleSolver(state, params=schur_params)

##############################################################################
# Set up forcing
##############################################################################
compressible_forcing = CompressibleEadyForcing(state, euler_poincare=False)

##############################################################################
# build time stepper
##############################################################################
stepper = Timestepper(state, advection_dict, linear_solver,
                      compressible_forcing)

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax)
