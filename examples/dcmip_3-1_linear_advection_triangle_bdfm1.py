from __future__ import absolute_import
from gusto import *
from firedrake import IcosahedralSphereMesh, ExtrudedMesh, Expression, \
    VectorFunctionSpace
from firedrake import par_loop, WRITE, READ
import numpy as np
import sys

dt = 10.
if '--running-tests' in sys.argv:
    nlayers = 2  # 2 horizontal layers
    refinements = 2  # number of horizontal cells = 20*(4^refinements)
    tmax = dt
else:
    nlayers = 10  # 10 horizontal layers
    refinements = 4  # number of horizontal cells = 20*(4^refinements)
    tmax = 3600.

# build surface mesh
a_ref = 6.37122e6
X = 125.0  # Reduced-size Earth reduction factor
a = a_ref/X
T_eq = 300.0  # Isothermal atmospheric temperature (K)
p_eq = 1000.0 * 100.0  # Reference surface pressure at the equator
d = 5000.0  # Width parameter for Theta'
lamda_c = 2.0*np.pi/3.0  # Longitudinal centerpoint of Theta'
phi_c = 0.0  # Latitudinal centerpoint of Theta' (equator)
deltaTheta = 1.0  # Maximum amplitude of Theta' (K)
L_z = 20000.0  # Vertical wave length of the Theta' perturbation
u_0 = 0.0  # Maximum amplitude of the zonal wind (m/s)

m = IcosahedralSphereMesh(radius=a,
                          refinement_level=refinements)

# build volume mesh
z_top = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=z_top/nlayers,
                    extrusion_type="radial")

# Space for initialising velocity
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
# Space for initialising reference profiles of theta and rho
W_CG1 = FunctionSpace(mesh, "CG", 1)

# Make a vertical direction for the linearised advection
k = Function(W_VectorCG1).interpolate(Expression(("x[0]/pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)","x[1]/pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)","x[2]/pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)")))

Omega = Function(W_VectorCG1).assign(0.0)

fieldlist = ['u','rho','theta']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(Verbose=True, dumpfreq=1, dirname='dcmip')
diagnostics = Diagnostics(*fieldlist)
parameters = CompressibleParameters()

state = CompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                          family="BDFM", k=k, Omega=Omega,
                          timestepping=timestepping,
                          output=output,
                          parameters=parameters,
                          diagnostics=diagnostics,
                          fieldlist=fieldlist)

# interpolate initial conditions
g = parameters.g
c_p = parameters.cp
N = parameters.N
p_0 = parameters.p_0
R_d = parameters.R_d
kappa = parameters.kappa

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

theta_b_expr = "(%(T_s)s)*pow(p_0/(%(p_s)s), kappa)*exp(N*N*%%(z)s/g)" % {'p_s': p_s_expr, 'T_s': T_s_expr} % string_expander

rho_expr = "(%(p)s)/(R_d*(%(T_b)s))" % {'p': p_expr, 'T_b': T_b_expr} % string_expander


thetab = Function(W_CG1)
thetab.interpolate(Expression(theta_b_expr, a=a, G=G, T_eq=T_eq, u_0=u_0, N=N, g=g, p_eq=p_eq, R_d=R_d, kappa=kappa, p_0=p_0))
theta_b = Function(state.V[2]).interpolate(thetab)

rhob = Function(W_CG1)
rhob.interpolate(Expression(rho_expr, a=a, G=G, T_eq=T_eq, u_0=u_0, N=N, g=g, p_eq=p_eq, R_d=R_d, kappa=kappa, p_0=p_0))
rho_b = Function(state.V[1])
rho_b.project(rhob)

theta_prime = Function(state.V[2])
dis_expr = "a*acos(sin(phi_c)*sin(%(lat)s) + cos(phi_c)*cos(%(lat)s)*cos(%(lon)s - lamda_c))"

theta_prime_expr = "dT*(d*d/(d*d + pow((%(dis)s), 2)))*sin(2*pi*%%(z)s/L_z)" % {'dis': dis_expr} % string_expander

thetaprime = Function(W_CG1)
thetaprime.interpolate(Expression(theta_prime_expr, dT=deltaTheta, d=d, L_z=L_z, a=a, phi_c=phi_c, lamda_c=lamda_c))
theta_prime.interpolate(thetaprime)

theta0.assign(theta_b + theta_prime)
rho0.assign(rho_b)

state.initialise([u0, rho0, theta0])
state.set_reference_profiles(rho_b, theta_b)

state.output.meanfields = {'rho':rho_b, 'theta':theta_b}

W_VectorDG0 = VectorFunctionSpace(mesh, "DG", 0)
# Build new extruded coordinate function space
zhat = Function(W_VectorDG0)

par_loop("""
double v0[3];
double v1[3];
double n[3];
double com[3];
double dot;
double norm;
norm = 0.0;
dot = 0.0;
// form "x1 - x0" and "x2 - x0" of cell base
for (int i=0; i<3; ++i) {
    v0[i] = coords[2][i] - coords[0][i];
    v1[i] = coords[4][i] - coords[0][i];
}

for (int i=0; i<3; ++i) {
    com[i] = 0.0;
}

// take cross-product to form normal vector
n[0] = v0[1] * v1[2] - v0[2] * v1[1];
n[1] = v0[2] * v1[0] - v0[0] * v1[2];
n[2] = v0[0] * v1[1] - v0[1] * v1[0];

// get (scaled) centre-of-mass of cell
for (int i=0; i<6; ++i) {
    com[0] += coords[i][0];
    com[1] += coords[i][1];
    com[2] += coords[i][2];
}

// is the normal pointing outwards or inwards w.r.t. origin?
for (int i=0; i<3; ++i) {
    dot += com[i]*n[i];
}

for (int i=0; i<3; ++i) {
    norm += n[i]*n[i];
}

// normalise normal vector and multiply by -1 if dot product was < 0
norm = sqrt(norm);
norm *= (dot < 0.0 ? -1.0 : 1.0);

for (int i=0; i<3; ++i) {
    normals[0][i] = n[i] / norm;
}
""", dx,
         {'normals': (zhat, WRITE),
          'coords': (mesh.coordinates, READ)})

state.zhat = zhat

# Set up advection schemes
advection_dict = {}
advection_dict["u"] = NoAdvection(state)
advection_dict["rho"] = LinearAdvection_V3(state, state.V[1], rho_b)
advection_dict["theta"] = LinearAdvection_Vt(state, state.V[2], theta_b)

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
compressible_forcing = CompressibleForcing(state, linear=True)

# build time stepper
stepper = Timestepper(state, advection_dict, linear_solver,
                      compressible_forcing)

stepper.run(t=0, tmax=tmax)
