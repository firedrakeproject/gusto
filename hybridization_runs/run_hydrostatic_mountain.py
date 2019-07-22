from gusto import *
import pandas as pd
from firedrake import (FunctionSpace, as_vector, VectorFunctionSpace,
                       PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       exp, pi, cos, Function, conditional, Mesh, sin, op2, sqrt,
                       errornorm, COMM_WORLD)


def minimum(f):
    fmin = op2.Global(1, [1000], dtype=float)
    op2.par_loop(op2.Kernel("""
static void minify(double *a, double *b) {
    a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
}
        """, "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
    return fmin.data[0]


res_dt = {10: 5.0, 20: 2.5, 30: 1.25, 40: 0.625}
resolutions = [10, 20, 30, 40]
tmax = 500.

L = 240000.

dumptime = 500.0   # write output every 500 sec.


for res in resolutions:

    dt = res_dt[res]
    dumpfreq = int(dumptime / dt)

    nlayers = res*20
    columns = res*12

    m = PeriodicIntervalMesh(columns, L)

    H = 50000.
    ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    Vc = VectorFunctionSpace(ext_mesh, "DG", 2)
    coord = SpatialCoordinate(ext_mesh)
    x = Function(Vc).interpolate(as_vector([coord[0], coord[1]]))
    a = 10000.
    xc = L/2.
    x, z = SpatialCoordinate(ext_mesh)
    hm = 1.
    zs = hm*a**2/((x-xc)**2 + a**2)

    dirname = 'h_mtn_hybridization_res%s' % res
    zh = 5000.
    xexpr = as_vector([x, conditional(z < zh, z + cos(0.5*pi*z/zh)**6*zs, z)])

    new_coords = Function(Vc).interpolate(xexpr)
    mesh = Mesh(new_coords)

    # sponge function
    W_DG = FunctionSpace(mesh, "DG", 2)
    x, z = SpatialCoordinate(mesh)
    zc = H-20000.
    mubar = 0.3/dt
    mu_top = conditional(z <= zc, 0.0, mubar*sin((pi/2.)*(z-zc)/(H-zc))**2)
    mu = Function(W_DG).interpolate(mu_top)
    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=dt, alpha=0.51)

    output = OutputParameters(dirname=dirname,
                              dumpfreq=dumpfreq,
                              dumplist=['u'],
                              perturbation_fields=['theta', 'rho'],
                              log_level='INFO')

    parameters = CompressibleParameters(g=9.80665, cp=1004.)
    diagnostics = Diagnostics(*fieldlist)
    diagnostic_fields = [CourantNumber(), VelocityZ(), HydrostaticImbalance()]

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  sponge_function=mu,
                  hydrostatic=True,
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  fieldlist=fieldlist,
                  diagnostic_fields=diagnostic_fields)

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = parameters.g
    p_0 = parameters.p_0
    c_p = parameters.cp
    R_d = parameters.R_d
    kappa = parameters.kappa

    # Hydrostatic case: Isothermal with T = 250
    Tsurf = 250.
    N = g/sqrt(c_p*Tsurf)

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    thetab = Tsurf*exp(N**2*z/g)
    theta_b = Function(Vt).interpolate(thetab)

    # Calculate hydrostatic Pi
    Pi = Function(Vr)
    rho_b = Function(Vr)

    piparams = {'ksp_type': 'gmres',
                'ksp_monitor_true_residual': None,
                'pc_type': 'python',
                'mat_type': 'matfree',
                'pc_python_type': 'gusto.VerticalHybridizationPC',
                # Vertical trace system is only coupled vertically in columns
                # block ILU is a direct solver!
                'vert_hybridization': {'ksp_type': 'preonly',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}}

    compressible_hydrostatic_balance(state, theta_b, rho_b, Pi,
                                     top=True, pi_boundary=0.5,
                                     params=piparams)


    p0 = minimum(Pi)
    compressible_hydrostatic_balance(state, theta_b, rho_b, Pi,
                                     top=True,
                                     params=piparams)
    p1 = minimum(Pi)
    alpha = 2.*(p1-p0)
    beta = p1-alpha
    pi_top = (1.-beta)/alpha
    compressible_hydrostatic_balance(state, theta_b, rho_b, Pi,
                                     top=True, pi_boundary=pi_top,
                                     solve_for_rho=True,
                                     params=piparams)

    theta0.assign(theta_b)
    rho0.assign(rho_b)
    u0.project(as_vector([20.0, 0.0]))
    remove_initial_w(u0, state.Vv)

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0)])
    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    # Set up advection schemes
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = SUPGAdvection(state, Vt, equation_form="advective")

    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
    advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

    # Set up linear solver
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.SCPC',
              # Velocity mass operator is singular in the hydrostatic case.
              # So for reconstruction, we eliminate rho into u
              'pc_sc_eliminate_fields': '1, 0',
              'condensed_field': {'ksp_type': 'fgmres',
                                  'ksp_rtol': 1.0e-8,
                                  'ksp_atol': 1.0e-8,
                                  'ksp_max_it': 100,
                                  'pc_type': 'gamg',
                                  'pc_gamg_sym_graph': True,
                                  'mg_levels': {'ksp_type': 'gmres',
                                                'ksp_max_it': 5,
                                                'pc_type': 'bjacobi',
                                                'sub_pc_type': 'ilu'}}}
    linear_solver = CompressibleSolver(state, solver_parameters=params,
                                       overwrite_solver_parameters=True)

    # Set up forcing
    compressible_forcing = CompressibleForcing(state)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            compressible_forcing)

    stepper.run(t=0, tmax=tmax, compute_hdiv_norms=True)

    t_hdivnorm = stepper.t_hdivnorm
    tarray = list(t_hdivnorm.keys())
    norms = list(t_hdivnorm.values())

    if COMM_WORLD.rank == 0:
        data = {"time": tarray,
                "hdiv_norm": norms}
        df = pd.DataFrame(data)
        df.to_csv(dirname + "_norms.csv", index=False, mode="w", header=True)
    
