from __future__ import absolute_import
from gusto import *
from firedrake import CubedSphereMesh, ExtrudedMesh, Expression
import numpy as np


def setup_dcmip(dirname):

    nlayers = 2         # 2 horizontal layers
    refinements = 3      # number of horizontal cells = 20*(4^refinements)

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
    u_0 = 20.0  # Maximum amplitude of the zonal wind (m/s)

    m = CubedSphereMesh(radius=a,
                        refinement_level=refinements,
                        degree=3)

    # build volume mesh
    z_top = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=z_top/nlayers,
                        extrusion_type="radial")

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=10.0)
    output = OutputParameters(Verbose=True, dumpfreq=1, dirname=dirname+"/dcmip", perturbation_fields=['theta', 'rho'])
    parameters = CompressibleParameters()

    state = State(mesh, vertical_degree=0, horizontal_degree=0,
                  family="RTCF",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    g = parameters.g
    c_p = parameters.cp
    N = parameters.N
    p_0 = parameters.p_0
    R_d = parameters.R_d
    kappa = parameters.kappa

    # interpolate initial conditions
    # Initial/current conditions
    theta0 = state.fields("theta")
    rho0 = state.fields("rho")
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Helper string processing
    string_expander = {'lat': "asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))", 'lon': "atan2(x[1], x[0])", 'r': "sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])", 'z': "(sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a)"}

    # Set up ICs

    G = g*g/(N*N*c_p)  # seems better than copy-pasting the string into code below...
    T_s_expr = "G + (T_eq - G)*exp(-((u_0*u_0*N*N)/(4*g*g))*(cos(2*%(lat)s) - 1.0))" % string_expander

    T_b_expr = "G*(1.0 - exp(N*N*%%(z)s/g)) + (%(T_s)s)*exp(N*N*%%(z)s/g)" % {'T_s': T_s_expr} % string_expander

    p_s_expr = "p_eq*exp(((u_0*u_0)/(4.0*G*R_d))*(cos(2*%%(lat)s) - 1.0))*pow((%(T_s)s)/T_eq, 1.0/kappa)" % {'T_s': T_s_expr} % string_expander

    p_expr = "(%(p_s)s)*pow((G/(%(T_s)s))*exp(-N*N*%%(z)s/g) + 1.0 - (G/(%(T_s)s)), 1.0/kappa)" % {'p_s': p_s_expr, 'T_s': T_s_expr} % string_expander

    theta_b_expr = "(%(T_s)s)*pow(p_0/(%(p_s)s), 1.0/kappa)*exp(N*N*%%(z)s/g)" % {'p_s': p_s_expr, 'T_s': T_s_expr} % string_expander

    rho_expr = "(%(p)s)/(R_d*(%(T_b)s))" % {'p': p_expr, 'T_b': T_b_expr} % string_expander

    theta_b = Function(Vt)
    theta_b.interpolate(Expression(theta_b_expr, a=a, G=G, T_eq=T_eq, u_0=u_0, N=N, g=g, p_eq=p_eq, R_d=R_d, kappa=kappa, p_0=p_0))

    rho_b = Function(Vr)
    rho_b.interpolate(Expression(rho_expr, a=a, G=G, T_eq=T_eq, u_0=u_0, N=N, g=g, p_eq=p_eq, R_d=R_d, kappa=kappa, p_0=p_0))

    theta_prime = Function(Vt)
    dis_expr = "a*acos(sin(phi_c)*sin(%(lat)s) + cos(phi_c)*cos(%(lat)s)*cos(%(lon)s - lamda_c))"

    theta_prime_expr = "dT*(d*d/(d*d + pow((%(dis)s), 2)))*sin(2*pi*%%(z)s/L_z)" % {'dis': dis_expr} % string_expander

    theta_prime.interpolate(Expression(theta_prime_expr, dT=deltaTheta, d=d, L_z=L_z, a=a, phi_c=phi_c, lamda_c=lamda_c))

    theta0.assign(theta_b + theta_prime)
    rho0.assign(rho_b)

    state.initialise({'rho': rho0, 'theta': theta0})
    state.set_reference_profiles({'rho': rho_b, 'theta': theta_b})

    # Set up advection schemes
    rhoeqn = LinearAdvection(state, Vr, qbar=rho_b, ibp="once", equation_form="continuity")
    thetaeqn = LinearAdvection(state, Vt, qbar=theta_b)
    advection_dict = {}
    advection_dict["u"] = NoAdvection(state, state.fields("u"))
    advection_dict["rho"] = ForwardEuler(state, rho0, rhoeqn)
    advection_dict["theta"] = ForwardEuler(state, theta0, thetaeqn)

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
              'fieldsplit_1_mg_levels_ksp_chebyshev_esteig': True,
              'fieldsplit_1_mg_levels_ksp_chebyshev_esteig_random': True,
              'fieldsplit_1_mg_levels_ksp_max_it': 5,
              'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
              'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

    linear_solver = CompressibleSolver(state, params=params)

    # Set up forcing
    compressible_forcing = CompressibleForcing(state, linear=True)

    # build time stepper
    stepper = Timestepper(state, advection_dict, linear_solver,
                          compressible_forcing)

    return stepper, timestepping.dt


def run_dcmip(dirname):

    stepper, dt = setup_dcmip(dirname)
    stepper.run(t=0, tmax=dt)


def test_dcmip_runs(tmpdir):

    dirname = str(tmpdir)
    run_dcmip(dirname)
