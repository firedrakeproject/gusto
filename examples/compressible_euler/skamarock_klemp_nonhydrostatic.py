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
    RungeKuttaFormulation, SubcyclingOptions
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
    theta_opts = SUPGOptions()
    subcycling_options = SubcyclingOptions(subcycle_by_courant=0.25)
    transported_fields = [
        SSPRK3(domain, "u", subcycling_options=subcycling_options),
        SSPRK3(
            domain, "rho", subcycling_options=subcycling_options,
            rk_formulation=RungeKuttaFormulation.linear
        ),
        SSPRK3(
            domain, "theta", subcycling_options=subcycling_options,
            options=theta_opts
        )
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho", advective_then_flux=True),
        DGUpwind(eqns, "theta", ibp=theta_opts.ibp)
    ]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

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

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = parameters.g

    x, z = SpatialCoordinate(mesh)

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta_b, rho_b)

    theta_pert = (
        deltaTheta * sin(pi*z/domain_height)
        / (1 + (x - domain_width/2)**2 / pert_width**2)
    )
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([wind_initial, 0.0]))

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
