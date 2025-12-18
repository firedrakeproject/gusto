"""
The dry rising bubble test from Bryan & Fritsch, 2002:
``A Benchmark Simulation for Moist Nonhydrostatic Numerical Models'', GMD.

This uses the lowest-order function spaces, with the recovered methods for
transporting the fields. The test also uses a non-periodic base mesh.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    IntervalMesh, ExtrudedMesh, SpatialCoordinate, conditional, cos, pi, sqrt,
    TestFunction, dx, TrialFunction, Constant, Function, as_vector,
    LinearVariationalProblem, LinearVariationalSolver
)
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    RecoverySpaces, BoundaryMethod, Perturbation, CompressibleParameters,
    CompressibleEulerEquations, CompressibleSolver,
    compressible_hydrostatic_balance
)

dry_bryan_fritsch_defaults = {
    'ncolumns': 100,
    'nlayers': 100,
    'dt': 2.0,
    'tmax': 1000.,
    'dumpfreq': 500,
    'dirname': 'dry_bryan_fritsch'
}


def dry_bryan_fritsch(
        ncolumns=dry_bryan_fritsch_defaults['ncolumns'],
        nlayers=dry_bryan_fritsch_defaults['nlayers'],
        dt=dry_bryan_fritsch_defaults['dt'],
        tmax=dry_bryan_fritsch_defaults['tmax'],
        dumpfreq=dry_bryan_fritsch_defaults['dumpfreq'],
        dirname=dry_bryan_fritsch_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    domain_width = 10000.     # domain width (m)
    domain_height = 10000.    # domain height (m)
    zc = 2000.                # vertical centre of bubble (m)
    rc = 2000.                # radius of bubble (m)
    Tdash = 2.0               # strength of temperature perturbation (K)
    Tsurf = 300.0             # background theta value (K)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 0
    u_eqn_type = 'vector_advection_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = IntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    domain = Domain(mesh, dt, "CG", element_order)

    # Equation
    params = CompressibleParameters(mesh)
    eqns = CompressibleEulerEquations(
        domain, params, u_transport_option=u_eqn_type,
        no_normal_flow_bc_ids=[1, 2]
    )

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True,
        dumplist=['rho']
    )
    diagnostic_fields = [Perturbation('theta')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes -- set up options for using recovery wrapper
    boundary_methods = {'DG': BoundaryMethod.taylor}

    recovery_spaces = RecoverySpaces(
        domain, boundary_methods, use_vector_spaces=True
    )

    u_opts = recovery_spaces.HDiv_options
    rho_opts = recovery_spaces.DG_options
    theta_opts = recovery_spaces.theta_options

    transported_fields = [
        SSPRK3(domain, "rho", options=rho_opts),
        SSPRK3(domain, "theta", options=theta_opts),
        SSPRK3(domain, "u", options=u_opts)
    ]

    transport_methods = [
        DGUpwind(eqns, field) for field in ["u", "rho", "theta"]
    ]

    # Linear solver
    linear_solver = CompressibleSolver(eqns, tau_values={'rho': 1.0, 'theta': 1.0})

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # spaces
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")
    x, z = SpatialCoordinate(mesh)

    # Define constant theta_e and water_t
    theta_b = Function(Vt).interpolate(Constant(Tsurf))

    # Set initial wind to be zero
    zero = Constant(0.0)
    u0.project(as_vector([zero, zero]))

    # Calculate hydrostatic fields
    compressible_hydrostatic_balance(eqns, theta_b, rho0, solve_for_rho=True)

    # make mean fields
    rho_b = Function(Vr).assign(rho0)

    # define perturbation
    xc = domain_width / 2
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    theta_pert = Function(Vt).interpolate(
        conditional(
            r > rc,
            0.0,
            Tdash * (cos(pi * r / (2.0 * rc))) ** 2
        )
    )

    # define initial theta
    theta0.interpolate(theta_b * (theta_pert / 300.0 + 1.0))

    # find perturbed rho
    gamma = TestFunction(Vr)
    rho_trial = TrialFunction(Vr)
    lhs = gamma * rho_trial * dx
    rhs = gamma * (rho_b * theta_b / theta0) * dx
    rho_problem = LinearVariationalProblem(lhs, rhs, rho0)
    rho_solver = LinearVariationalSolver(rho_problem)
    rho_solver.solve()

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
        default=dry_bryan_fritsch_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=dry_bryan_fritsch_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=dry_bryan_fritsch_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=dry_bryan_fritsch_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=dry_bryan_fritsch_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=dry_bryan_fritsch_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    dry_bryan_fritsch(**vars(args))
