"""
This example uses the non-linear compressible Euler equations to solve the
vertical slice gravity wave test case of Skamarock and Klemp, 1994:
``Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique'',
MWR.

The domain is smaller than the "hydrostatic" gravity wave test, so that there
is difference between the hydrostatic and non-hydrostatic solutions. The test
can be run with and without a hydrostatic switch.

Potential temperature is transported using SUPG, and the degree 1 elements are
used.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
from firedrake import (
    as_vector, SpatialCoordinate, PeriodicIntervalMesh, ExtrudedMesh, exp, sin,
    Function, pi
)
from gusto import (
    OutputParameters, Perturbation, CompressibleParameters,
    CompressibleEulerEquations, Model,
    compressible_hydrostatic_balance, CompressibleSolver
)

skamarock_klemp_nonhydrostatic_defaults = {
    'ncolumns': 150,
    'nlayers': 10,
    'dt': 6.0,
    'tmax': 3000.,
    'dumpfreq': 250,
    'dirname': 'skamarock_klemp_nonhydrostatic',
    'hydrostatic': False
}


def skamarock_klemp_nonhydrostatic(
        ncolumns=skamarock_klemp_nonhydrostatic_defaults['ncolumns'],
        nlayers=skamarock_klemp_nonhydrostatic_defaults['nlayers'],
        dt=skamarock_klemp_nonhydrostatic_defaults['dt'],
        tmax=skamarock_klemp_nonhydrostatic_defaults['tmax'],
        dumpfreq=skamarock_klemp_nonhydrostatic_defaults['dumpfreq'],
        dirname=skamarock_klemp_nonhydrostatic_defaults['dirname'],
        hydrostatic=skamarock_klemp_nonhydrostatic_defaults['hydrostatic']
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
    # Set up model
    # ------------------------------------------------------------------------ #

    # Domain -- 3D volume mesh
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)

    # Equation
    parameters = CompressibleParameters(mesh)
    eqns = CompressibleEulerEquations

    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, pddumpfreq=dumpfreq,
        dump_vtus=False, dump_nc=True,
    )

    diagnostic_fields = [Perturbation('theta')]

    model = Model(mesh, dt, parameters, eqns, output,
                  linear_solver=CompressibleSolver,
                  diagnostic_fields=diagnostic_fields)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = model.stepper.fields("u")
    rho0 = model.stepper.fields("rho")
    theta0 = model.stepper.fields("theta")

    # spaces
    Vt = model.domain.spaces("theta")
    Vr = model.domain.spaces("DG")

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = parameters.g

    x, z = SpatialCoordinate(mesh)

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(model.eqns, theta_b, rho_b)

    theta_pert = (
        deltaTheta * sin(pi*z/domain_height)
        / (1 + (x - domain_width/2)**2 / pert_width**2)
    )
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([wind_initial, 0.0]))

    model.stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    model.run(t=0, tmax=tmax)

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
    parser.add_argument(
        '--hydrostatic',
        help=(
            "Whether to use the hydrostatic switch to emulate the "
            + "hydrostatic equations. Otherwise use the full non-hydrostatic"
            + "equations."
        ),
        action="store_true",
        default=skamarock_klemp_nonhydrostatic_defaults['hydrostatic']
    )
    args, unknown = parser.parse_known_args()

    skamarock_klemp_nonhydrostatic(**vars(args))
