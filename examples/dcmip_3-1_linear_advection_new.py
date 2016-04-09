from dcore import *
from firedrake import IcosahedralSphereMesh, ExtrudedMesh, Expression, \
    VectorFunctionSpace
from firedrake import exp,asin
import numpy as np

nlayers = 10 #10 horizontal layers
refinements = 3 # number of horizontal cells = 20*(4^refinements)

#build surface mesh
a_ref = 6.37122e6
X = 125.0  # Reduced-size Earth reduction factor
a = a_ref/X
g = 9.81
N = 0.01  # Brunt-Vaisala frequency (1/s)
p_0 = 1000.0 * 100.0  # Reference pressure (Pa, not hPa)
c_p = 1004.5  # SHC of dry air at constant pressure (J/kg/K)
R_d = 287.0  # Gas constant for dry air (J/kg/K)
kappa = 2.0/7.0  # R_d/c_p
T_eq = 300.0  # Isothermal atmospheric temperature (K)
p_eq = 1000.0 * 100.0  # Reference surface pressure at the equator
d = 5000.0  # Width parameter for Theta'
lamda_c = 2.0*np.pi/3.0  # Longitudinal centerpoint of Theta'
phi_c = 0.0  # Latitudinal centerpoint of Theta' (equator)
deltaTheta = 1.0  # Maximum amplitude of Theta' (K)
L_z = 20000.0  # Vertical wave length of the Theta' perturbation
u_0 = 20.0  # Maximum amplitude of the zonal wind (m/s)

m = IcosahedralSphereMesh(radius = a,
                          refinement_level = refinements)

#build volume mesh
z_top = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers = nlayers, layer_height = z_top/nlayers,
                    extrusion_type="radial")

# Space for initialising velocity
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)

Omega = Function(W_VectorCG1).assign(0.0)

#Create polar coordinates
z = Function(W_CG1).interpolate(Expression("sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a",a=a)) #Since we use a CG1 field, this is constant on layers
lat = Function(W_CG1).interpolate(Expression("asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))"))
lon = Function(W_CG1).interpolate(Expression("atan2(x[1], x[0])"))

state = State(mesh,vertical_degree = 0, horizontal_degree = 0,
              family = "BDM",
              dt = 10.0,
              alpha = 0.5,
              g = g,
              cp = c_p,
              R_d = R_d,
              p_0 = p_0,
              z=z,
              Omega=Omega,
              Verbose=True, dumpfreq=1)

# Initial conditions
u0, theta0, rho0 = Function(state.V[0]), Function(state.V[2]), Function(state.V[1])

#Initial conditions without u0
#Surface temperature
G = g**2/N**2/c_p
Ts = Function(W_CG1).assign(G) 
#Background temperature
Tb = Function(W_CG1).interpolate(G*(1-exp(N**2*z/g) + Ts*exp(N**2*z/g)))
#surface pressure
ps = Function(W_CG1).interpolate(p_eq*(Ts/T_eq)**(1.0/kappa))
#Background pressure
p = Function(W_CG1).interpolate(ps*(G/Ts*exp(-N**2*z/g)+1-G/Ts)**(1/kappa))
#Background potential temperature
thetab = Function(W_CG1).interpolate(Ts*(p_0/ps)**kappa*exp(N**2*z/g))
rhob = Function(W_CG1).interpolate(p/R_d/Tb)

theta_b = Function(state.V[2]).interpolate(thetab)
rho_b = Function(state.V[1]).project(rhob)

theta0.assign(theta_b)
rho0.assign(rho_b)

state.initialise(u0, rho0, theta0)
state.set_reference_profiles(rho_b, theta_b)

#Set up advection schemes
advection_list = []
velocity_advection = NoAdvection(state)
advection_list.append((velocity_advection, 0))
rho_advection = LinearAdvection_V3(state, state.V[1], rho_b)
advection_list.append((rho_advection, 1))
theta_advection = LinearAdvection_Vt(state, state.V[2], theta_b)
advection_list.append((theta_advection, 2))

#Set up linear solver
schur_params={'pc_type': 'fieldsplit',
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
        'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'
}

linear_solver = CompressibleSolver(state, alpha = 0.5, params = schur_params)

#Set up forcing
compressible_forcing = CompressibleForcing(state)

#build time stepper
stepper = Timestepper(state, advection_list, linear_solver,
                      compressible_forcing)

stepper.run(t = 0, tmax = 3600.0)

