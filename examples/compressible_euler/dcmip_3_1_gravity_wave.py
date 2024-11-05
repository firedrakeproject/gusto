"""
The non-orographic gravity wave test case (3-1) from the DCMIP test case
document of Ullrich et al, 2012:
``Dynamical core model intercomparison project (DCMIP) test case document''.

This uses a cubed-sphere mesh, the degree 1 finite element spaces and tests
substepping the transport schemes.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    ExtrudedMesh, Function, SpatialCoordinate, as_vector,
    exp, acos, cos, sin, pi
)
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, SUPGOptions, lonlatr_from_xyz, CompressibleParameters,
    CompressibleEulerEquations, CompressibleSolver, ZonalComponent,
    compressible_hydrostatic_balance, Perturbation, GeneralCubedSphereMesh,
    Timestepper, IMEX_SSP3, split_continuity_form, explicit, implicit,
    time_derivative, transport, IMEXRungeKutta
)
import numpy as np

dcmip_3_1_gravity_wave_defaults = {
    'ncells_per_edge': 8,
    'nlayers': 10,
    'dt': 50.0,
    'tmax': 3600.,
    'dumpfreq': 9,
    'dirname': 'dcmip_3_1_gravity_wave'
}


def dcmip_3_1_gravity_wave(
        ncells_per_edge=dcmip_3_1_gravity_wave_defaults['ncells_per_edge'],
        nlayers=dcmip_3_1_gravity_wave_defaults['nlayers'],
        dt=dcmip_3_1_gravity_wave_defaults['dt'],
        tmax=dcmip_3_1_gravity_wave_defaults['tmax'],
        dumpfreq=dcmip_3_1_gravity_wave_defaults['dumpfreq'],
        dirname=dcmip_3_1_gravity_wave_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    parameters = CompressibleParameters()
    a_ref = 6.37122e6               # Radius of the Earth (m)
    X = 125.0                       # Reduced-size Earth reduction factor
    a = a_ref/X                     # Scaled radius of planet (m)
    g = parameters.g                # Acceleration due to gravity (m/s^2)
    N = 0.01                        # Brunt-Vaisala frequency (1/s)
    p_0 = parameters.p_0            # Reference pressure (Pa, not hPa)
    c_p = parameters.cp             # SHC of dry air at constant pressure (J/kg/K)
    R_d = parameters.R_d            # Gas constant for dry air (J/kg/K)
    kappa = parameters.kappa        # R_d/c_p
    T_eq = 300.0                    # Isothermal atmospheric temperature (K)
    p_eq = 1000.0 * 100.0           # Reference surface pressure at the equator
    u_max = 20.0                    # Maximum amplitude of the zonal wind (m/s)
    d = 5000.0                      # Width parameter for Theta'
    lamda_c = 2.0*pi/3.0            # Longitudinal centerpoint of Theta'
    phi_c = 0.0                     # Latitudinal centerpoint of Theta' (equator)
    deltaTheta = 1.0                # Maximum amplitude of Theta' (K)
    L_z = 20000.0                   # Vertical wave length of the Theta' perturb.
    z_top = 1.0e4                   # Height position of the model top

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1
    u_eqn_type = 'vector_invariant_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = GeneralCubedSphereMesh(a, ncells_per_edge, degree=2)
    mesh = ExtrudedMesh(
        base_mesh, nlayers, layer_height=z_top/nlayers,
        extrusion_type="radial"
    )
    domain = Domain(mesh, dt, "RTCF", element_order)

    # Equation
    eqns = CompressibleEulerEquations(
        domain, parameters, u_transport_option=u_eqn_type
    )
    print("Opt Cores:", eqns.X.function_space().dim()/50000.)
    eqns = split_continuity_form(eqns)
    eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
    #eqns.label_terms(lambda t: not any(t.has_label(time_derivative, horizontal)), implicit)
    eqns.label_terms(lambda t: t.has_label(transport), explicit)
    # eqns.label_terms(lambda t: t.has_label(transport) and t.has_label(vertical), implicit)
    # eqns.label_terms(lambda t: t.has_label(transport) and not any(t.has_label(horizontal, vertical)), explicit)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True
    )
    diagnostic_fields = [
        Perturbation('theta'), Perturbation('rho'), ZonalComponent('u'),
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    # transported_fields = [
    #     TrapeziumRule(domain, "u"),
    #     SSPRK3(domain, "rho", fixed_subcycles=2),
    #     SSPRK3(domain, "theta", options=SUPGOptions(), fixed_subcycles=2)
    # ]
    transport_methods = [
        DGUpwind(eqns, field) for field in ["u", "rho", "theta"]
    ]

    nl_solver_parameters = {
    "snes_converged_reason": None,
    "snes_lag_preconditioner_persists":None,
    "snes_lag_preconditioner":-2,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-5,
    "ksp_rtol": 1e-5,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_star_sub_sub_pc_type": "lu",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    "assembled_pc_star_sub_sub_pc_factor_reuse_ordering": None,
    "assembled_pc_star_sub_sub_pc_factor_reuse_fill": None,
    "assembled_pc_star_sub_sub_pc_factor_fill": 1.2}


    # IMEX time stepper

    butcher_imp = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, 1.]])
    butcher_exp = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 1.]])
    scheme = IMEXRungeKutta(domain, butcher_imp, butcher_exp, solver_parameters=nl_solver_parameters)
    stepper = Timestepper(eqns, scheme, io, transport_methods)
    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields('u')
    theta0 = stepper.fields('theta')
    rho0 = stepper.fields('rho')

    # spaces
    Vr = domain.spaces("DG")

    x, y, z = SpatialCoordinate(mesh)
    lon, lat, r = lonlatr_from_xyz(x, y, z)
    h = r - a

    # Initial conditions with u0
    uexpr = as_vector([-u_max*y/a, u_max*x/a, 0.0])

    # Surface temperature
    G = g**2/(N**2*c_p)
    Ts_expr = (
        G + (T_eq - G) * exp(-(u_max*N**2/(4*g*g)) * u_max*(cos(2.0*lat)-1.0))
    )

    # Surface pressure
    ps_expr = (
        p_eq * exp((u_max/(4.0*G*R_d)) * u_max*(cos(2.0*lat)-1.0))
        * (Ts_expr / T_eq)**(1.0/kappa)
    )

    # Background pressure
    p_expr = ps_expr*(1 + G/Ts_expr*(exp(-N**2*h/g)-1))**(1.0/kappa)
    p = Function(Vr).interpolate(p_expr)

    # Background temperature
    Tb_expr = G*(1 - exp(N**2*h/g)) + Ts_expr*exp(N**2*h/g)

    # Background potential temperature
    thetab_expr = Tb_expr*(p_0/p)**kappa
    theta_b = Function(theta0.function_space()).interpolate(thetab_expr)
    rho_b = Function(rho0.function_space())
    sin_tmp = sin(lat) * sin(phi_c)
    cos_tmp = cos(lat) * cos(phi_c)
    l = a*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))
    s = (d**2)/(d**2 + l**2)
    theta_pert = deltaTheta*s*sin(2*pi*h/L_z)

    # Compute the balanced density
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, top=False, exner_boundary=(p/p_0)**kappa
    )

    u0.project(uexpr)
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)

    stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    # Run!
    stepper.run(t=0, tmax=tmax)

# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ncells_per_edge',
        help="The number of cells per panel edge of the cubed-sphere.",
        type=int,
        default=dcmip_3_1_gravity_wave_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=dcmip_3_1_gravity_wave_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=dcmip_3_1_gravity_wave_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=dcmip_3_1_gravity_wave_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=dcmip_3_1_gravity_wave_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=dcmip_3_1_gravity_wave_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    dcmip_3_1_gravity_wave(**vars(args))
