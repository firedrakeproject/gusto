from gusto import *
from firedrake import CubedSphereMesh, ExtrudedMesh, Expression, \
    VectorFunctionSpace
from firedrake import exp, acos, cos, sin
import numpy as np

nlayers = 12  # Number of horizontal layers (was 12)
refinements = 4  # number of horizontal cells = 20*(4^refinements) (was 4)

# build surface mesh
a_ref = 6.37122e6
X = 125.0  # Reduced-size Earth reduction factor
a = a_ref/X
g = 9.810616
N = 0.01  # Brunt-Vaisala frequency (1/s)
p_0 = 1000.0 * 100.0  # Reference pressure (Pa, not hPa)
c_p = 1004.5  # SHC of dry air at constant pressure (J/kg/K)
R_d = 287.0  # Gas constant for dry air (J/kg/K)
kappa = 2.0/7.0  # R_d/c_p
T_eq = 300.0  # Isothermal atmospheric temperature (K)
p_eq = 1000.0 * 100.0  # Reference surface pressure at the equator
u_0 = 20.0  # Maximum amplitude of the zonal wind (m/s)

d = 5000.0  # Width parameter for Theta'
lamda_c = 2.0*np.pi/3.0  # Longitudinal centerpoint of Theta'
phi_c = 0.0  # Latitudinal centerpoint of Theta' (equator)
deltaTheta = 1.0  # Maximum amplitude of Theta' (K)
L_z = 20000.0  # Vertical wave length of the Theta' perturbation

m = CubedSphereMesh(radius=a,
                    refinement_level=refinements,
                    degree=2)

# build volume mesh
z_top = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=z_top/nlayers,
                    extrusion_type="radial")

# Space for initialising velocity
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)

Omega = Function(W_VectorCG1).assign(0.0)

# Create polar coordinates
z = Function(W_CG1).interpolate(Expression("sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a",a=a))  # Since we use a CG1 field, this is constant on layers
lat = Function(W_CG1).interpolate(Expression("asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))"))
lon = Function(W_CG1).interpolate(Expression("atan2(x[1], x[0])"))

k = Function(W_VectorCG1).interpolate(
    Expression(("x[0]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
               "x[1]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                "x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])")))

fieldlist = ['u','rho','theta']
timestepping = TimesteppingParameters(dt=10.0, maxk=4, maxi=1)
output = OutputParameters(Verbose=True, dumpfreq=1, dirname='dcmip_new')
parameters = CompressibleParameters()

state = CompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                          family="RTCF", k=None, z=z,
                          timestepping=timestepping,
                          output=output,
                          parameters=parameters,
                          fieldlist=fieldlist,
                          on_sphere=True)

Vtdg_elt = BrokenElement(state.V_elt[2])
Vtdg = FunctionSpace(mesh, Vtdg_elt)

# Initial conditions
u0, theta0, rho0 = Function(state.V[0]), Function(state.V[2]), Function(state.V[1])

# Initial conditions without u0
# Surface temperature
G = g**2/(N**2*c_p)
Ts = Function(W_CG1).assign(T_eq)

# surface pressure
psexp = p_eq*((Ts/T_eq)**(1.0/kappa))

ps = Function(W_CG1).interpolate(psexp)

# Background pressure
pexp = ps*(1 + G/Ts*(exp(-N**2*z/g)-1))**(1.0/kappa)

p = Function(W_CG1).interpolate(pexp)

# Background temperature
# Tbexp = Ts*(p/ps)**kappa/(Ts/G*((p/ps)**kappa - 1) + 1)
Tbexp = G*(1 - exp(N**2*z/g)) + Ts*exp(N**2*z/g)

Tb = Function(W_CG1).interpolate(Tbexp)

# Background potential temperature
thetabexp = Tb*(p_0/p)**kappa

thetab = Function(W_CG1).interpolate(thetabexp)

theta_b = Function(state.V[2]).interpolate(thetab)
rho_b = Function(state.V[1])

sin_tmp = sin(lat) * sin(phi_c)
cos_tmp = cos(lat) * cos(phi_c)

r = a*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))

s = (d**2)/(d**2 + r**2)

theta_pert = deltaTheta*s*sin(2*np.pi*z/L_z)

theta0.interpolate(theta_b)
# Compute the balanced density
compressible_hydrostatic_balance(state, theta_b, rho_b, top=False,
                                 pi_boundary=(p/p_0)**kappa)
theta0.interpolate(theta_pert)
theta0 += theta_b
rho0.assign(rho_b)

state.initialise([u0, rho0, theta0])
state.set_reference_profiles(rho_b, theta_b)
state.output.meanfields = {'rho':rho_b, 'theta':theta_b}

# Set up advection schemes
advection_list = []
velocity_advection = EulerPoincareForm(state, state.V[0])
advection_list.append((velocity_advection, 0))
rho_advection = DGAdvection(state, state.V[1], continuity=True)
advection_list.append((rho_advection, 1))
theta_advection = EmbeddedDGAdvection(state, state.V[2],
                                      Vdg=Vtdg, continuity=False)
advection_list.append((theta_advection, 2))

# Set up linear solver
schur_amg_params = {'pc_type': 'fieldsplit',
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

linear_solver = CompressibleSolver(state, params=schur_amg_params)

# Set up forcing
compressible_forcing = CompressibleForcing(state)

# build time stepper
stepper = Timestepper(state, advection_list, linear_solver,
                      compressible_forcing)

stepper.run(t=0, tmax=3600.0)
