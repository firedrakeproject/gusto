from __future__ import absolute_import
from dcore import *
from firedrake import IcosahedralSphereMesh, ExtrudedMesh, Expression, \
    VectorFunctionSpace
import numpy as np

nlayers = 10         # 10 horizontal layers
refinements = 5      # number of horizontal cells = 20*(4^refinements)

# build surface mesh
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

m = IcosahedralSphereMesh(radius=a,
                          refinement_level=refinements)

# build volume mesh
z_top = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=z_top/nlayers,
                    extrusion_type="radial")

# Space for initialising velocity
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)

# Make a vertical direction for the linearised advection
k = Function(W_VectorCG1).interpolate(Expression(("x[0]/pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)","x[1]/pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)","x[2]/pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)")))

Omega = Function(W_VectorCG1).assign(0.0)

state = Compressible3DState(mesh,
                            family="BDFM",
                            dt=10.0,
                            alpha=0.5,
                            g=g,
                            cp=c_p,
                            R_d=R_d,
                            p_0=p_0,
                            k=k,
                            Omega=Omega,
                            Verbose=True, dumpfreq=1)

state.fieldlist = ('u', 'rho', 'theta')

# interpolate initial conditions
# Initial/current conditions
u0, theta0, rho0 = Function(state.V[0]), Function(state.V[2]), Function(state.V[1])

# Helper string processing
string_expander = {'lat': "asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))", 'lon': "atan2(x[1], x[0])", 'r': "sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])", 'z': "(sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a)"}

# Set up ICs
zonal_expr = "u_0*cos(%(lat)s)*(%(r)s/a)"
sph_vel_expander = {'zonal': zonal_expr}
cartesian_u_expr_hz = "-(%(zonal)s)*sin(%%(lon)s)" % sph_vel_expander % string_expander
cartesian_v_expr_hz = "(%(zonal)s)*cos(%%(lon)s)" % sph_vel_expander % string_expander
cartesian_w_expr_hz = "0.0"
u0_hz_expr = Expression((cartesian_u_expr_hz, cartesian_v_expr_hz, cartesian_w_expr_hz), a=a, u_0=u_0)
u0_hz = Function(W_VectorCG1).interpolate(u0_hz_expr)

Project_horizontal = False
if(Project_horizontal):
    # project into h-only at first to avoid any spurious vertical velocity issues
    u0_h = Function(W2h)

    W2h_trial = TrialFunction(W2h)
    W2h_test = TestFunction(W2h)
    solve(dot(W2h_trial, W2h_test)*dx == dot(u0_hz, W2h_test)*dx, u0_h)

    velocity_bcs = [DirichletBC(W2, 0.0, "bottom"), DirichletBC(W2, 0.0, "top")]

    W2_trial = TrialFunction(W2)
    W2_test = TestFunction(W2)
    solve(dot(W2_trial, W2_test)*dx == dot(u0_h, W2_test)*dx, u0, bcs=velocity_bcs)

G = g*g/(N*N*c_p)  # seems better than copy-pasting the string into code below...
T_s_expr = "G + (T_eq - G)*exp(-((u_0*u_0*N*N)/(4*g*g))*(cos(2*%(lat)s) - 1.0))" % string_expander

T_b_expr = "G*(1.0 - exp(N*N*%%(z)s/g)) + (%(T_s)s)*exp(N*N*%%(z)s/g)" % {'T_s': T_s_expr} % string_expander

p_s_expr = "p_eq*exp(((u_0*u_0)/(4.0*G*R_d))*(cos(2*%%(lat)s) - 1.0))*pow((%(T_s)s)/T_eq, 1.0/kappa)" % {'T_s': T_s_expr} % string_expander

p_expr = "(%(p_s)s)*pow((G/(%(T_s)s))*exp(-N*N*%%(z)s/g) + 1.0 - (G/(%(T_s)s)), 1.0/kappa)" % {'p_s': p_s_expr, 'T_s': T_s_expr} % string_expander

theta_b_expr = "(%(T_s)s)*pow(p_0/(%(p_s)s), 1.0/kappa)*exp(N*N*%%(z)s/g)" % {'p_s': p_s_expr, 'T_s': T_s_expr} % string_expander

rho_expr = "(%(p)s)/(R_d*(%(T_b)s))" % {'p': p_expr, 'T_b': T_b_expr} % string_expander

theta_b = Function(state.V[2])
theta_b.interpolate(Expression(theta_b_expr, a=a, G=G, T_eq=T_eq, u_0=u_0, N=N, g=g, p_eq=p_eq, R_d=R_d, kappa=kappa, p_0=p_0))

rho_b = Function(state.V[1])
rho_b.interpolate(Expression(rho_expr, a=a, G=G, T_eq=T_eq, u_0=u_0, N=N, g=g, p_eq=p_eq, R_d=R_d, kappa=kappa, p_0=p_0))

theta_prime = Function(state.V[2])
dis_expr = "a*acos(sin(phi_c)*sin(%(lat)s) + cos(phi_c)*cos(%(lat)s)*cos(%(lon)s - lamda_c))"

theta_prime_expr = "dT*(d*d/(d*d + pow((%(dis)s), 2)))*sin(2*pi*%%(z)s/L_z)" % {'dis': dis_expr} % string_expander

theta_prime.interpolate(Expression(theta_prime_expr, dT=deltaTheta, d=d, L_z=L_z, a=a, phi_c=phi_c, lamda_c=lamda_c))

theta0.assign(theta_b + theta_prime)
rho0.assign(rho_b)

state.initialise([u0, rho0, theta0])
state.set_reference_profiles(rho_b, theta_b)

# Set up advection schemes
advection_list = []
velocity_advection = NoAdvection(state)
advection_list.append((velocity_advection, 0))
rho_advection = LinearAdvection_V3(state, rho_b)
advection_list.append((rho_advection, 1))
theta_advection = LinearAdvection_Vt(state, k, theta_b)
advection_list.append((theta_advection, 2))

# Set up linear solver
params = {'pc_type': 'fieldsplit',
          'pc_fieldsplit_type': 'schur',
          'ksp_type': 'gmres',
          'ksp_monitor_true_residual': True,
          'ksp_max_it': 100,
          'ksp_gmres_restart': 50,
          'pc_fieldsplit_schur_fact_type': 'FULL',
          'pc_fieldsplit_schur_precondition': 'selfp',
          'fieldsplit_0_ksp_type': 'richardson',
          'fieldsplit_0_ksp_max_it': 2,
          'fieldsplit_0_pc_type': 'bjacobi',
          'fieldsplit_0_sub_pc_type': 'ilu',
          'fieldsplit_1_ksp_type': 'richardson',
          'fieldsplit_1_ksp_max_it': 2,
          "fieldsplit_1_ksp_monitor_true_residual": True,
          'fieldsplit_1_pc_type': 'gamg',
          'fieldsplit_1_pc_gamg_sym_graph': True,
          'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
          'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
          'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
          'fieldsplit_1_mg_levels_ksp_max_it': 5,
          'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
          'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

linear_solver = CompressibleSolver(state, params=params)

# Set up forcing
compressible_forcing = CompressibleForcing(state)

# build time stepper
stepper = Timestepper(state, advection_list, linear_solver,
                      compressible_forcing)

stepper.run(t=0, tmax=3600.0)
