"""
This example uses the non-linear compressible Euler equations to solve the
vertical slice gravity wave test case of Skamarock and Klemp, 1994:
``Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique'',
MWR.

Potential temperature is transported using SUPG, and the degree 1 elements are
used.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
import itertools
from firedrake import (
    as_vector, SpatialCoordinate, PeriodicIntervalMesh, ExtrudedMesh, exp, sin,
    Function, pi, COMM_WORLD
)
import numpy as np
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    SUPGOptions, CourantNumber, Perturbation, Gradient,
    CompressibleParameters, CompressibleEulerEquations, CompressibleSolver,
    compressible_hydrostatic_balance, logger, RichardsonNumber,
    time_derivative, transport, implicit, explicit, split_continuity_form,
    IMEXRungeKutta,  Timestepper, thermodynamics, eos_form, eos_mass,
    ImplicitMidpoint, RungeKuttaFormulation
)

skamarock_klemp_nonhydrostatic_defaults = {
    'ncolumns': 150,
    'nlayers': 10,
    'dt': 6.0,
    'tmax': 3600.,
    'dumpfreq': 300,
    'dirname': 'skamarock_klemp_nonhydrostatic'
}


def skamarock_klemp_nonhydrostatic(
        ncolumns=skamarock_klemp_nonhydrostatic_defaults['ncolumns'],
        nlayers=skamarock_klemp_nonhydrostatic_defaults['nlayers'],
        dt=skamarock_klemp_nonhydrostatic_defaults['dt'],
        tmax=skamarock_klemp_nonhydrostatic_defaults['tmax'],
        dumpfreq=skamarock_klemp_nonhydrostatic_defaults['dumpfreq'],
        dirname=skamarock_klemp_nonhydrostatic_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    domain_width = 3.0e5      # Width of domain (m)
    domain_height = 1.0e4     # Height of domain (m)
    Tsurf = 300.              # Temperature at surface (K)
    wind_initial = 20.        # Initial wind in x direction (m/s)
    pert_width = 5.0e3        # Width parameter of perturbation (m)
    deltaTheta = 1.0e-2       # Magnitude of theta perturbation (K)
    N = 0.01                  # Brunt-Vaisala frequency (1/s)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain -- 3D volume mesh
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    domain = Domain(mesh, dt, "CG", element_order)

    # Equation
    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(domain, parameters)
    # Check number of optimal cores
    # print("Opt Cores:", eqns.X.function_space().dim()/50000.)
    #eqns = split_continuity_form(eqns)
    eqns.label_terms(lambda t: not any(t.has_label(time_derivative, eos_form, eos_mass)), implicit)
    # eqns.label_terms(lambda t: t.has_label(transport), explicit)

    # I/O
    points_x = np.linspace(0., domain_width, 100)
    points_z = [domain_height/2.]
    points = np.array([p for p in itertools.product(points_x, points_z)])

    # Dumping point data using legacy PointDataOutput is not supported in parallel
    if COMM_WORLD.size == 1:
        output = OutputParameters(
            dirname=dirname, dumpfreq=dumpfreq, pddumpfreq=dumpfreq,
            dump_vtus=True, dump_nc=False,
            point_data=[('theta_perturbation', points)],
        )
    else:
        logger.warning(
            'Dumping point data using legacy PointDataOutput is not'
            ' supported in parallel\nDisabling PointDataOutput'
        )
        output = OutputParameters(
            dirname=dirname, dumpfreq=dumpfreq, pddumpfreq=dumpfreq,
            dump_vtus=True, dump_nc=True,
        )

    diagnostic_fields = [
        CourantNumber(), Gradient('u'), Perturbation('theta'),
        Gradient('theta_perturbation'), Perturbation('rho'),
        RichardsonNumber('theta', parameters.g/Tsurf), Gradient('theta')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    # theta_opts = SUPGOptions()
    # transported_fields = [
    #     TrapeziumRule(domain, "u"),
    #     SSPRK3(domain, "rho"),
    #     SSPRK3(domain, "theta", options=theta_opts)
    # ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho"),
        DGUpwind(eqns, "theta")
    ]

    nl_solver_parameters = {
    "snes_converged_reason": None,
    "snes_lag_preconditioner_persists":None,
    "snes_lag_preconditioner":-2, 
    "mat_type": "matfree",
    "ksp_type": "gmres",
    'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
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

    butcher_imp =np.array([[0.5], [1.]])
    butcher_exp = np.array([[0.5], [1.]])
    scheme = IMEXRungeKutta(domain, butcher_imp, butcher_exp, nonlinear_solver_parameters=nl_solver_parameters)
    #scheme = TrapeziumRule(domain, solver_parameters=nl_solver_parameters)
    #Time stepper
    scheme=ImplicitMidpoint(domain, solver_parameters=nl_solver_parameters, rk_formulation=RungeKuttaFormulation.predictor,)
    stepper = Timestepper(eqns, scheme, io, transport_methods)


    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")
    exner0 = stepper.fields("exner")

    # spaces
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = parameters.g

    x, z = SpatialCoordinate(mesh)

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta_b, rho_b, exner0)

    theta_pert = (
        deltaTheta * sin(pi*z/domain_height)
        / (1 + (x - domain_width/2)**2 / pert_width**2)
    )
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([wind_initial, 0.0]))

    #exner0.interpolate(thermodynamics.exner_pressure(parameters, rho0, theta0))

    stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

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
        '--ncolumns',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=skamarock_klemp_nonhydrostatic_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=skamarock_klemp_nonhydrostatic_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=skamarock_klemp_nonhydrostatic_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=skamarock_klemp_nonhydrostatic_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=skamarock_klemp_nonhydrostatic_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=skamarock_klemp_nonhydrostatic_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    skamarock_klemp_nonhydrostatic(**vars(args))
