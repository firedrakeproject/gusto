"""
The falling cold density current test of Straka et al, 1993:
``Numerical solutions of a non‐linear density current: A benchmark solution and
comparisons'', MiF.

Diffusion is included in the velocity and potential temperature equations. The
degree 1 finite elements are used in this configuration.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, Constant, pi, cos,
    Function, sqrt, conditional, as_vector
)
from gusto import (
    OutputParameters, CourantNumber, Perturbation,
    DiffusionParameters,
    CompressibleParameters, CompressibleEulerEquations, CompressibleSolver,
    compressible_hydrostatic_balance, Model
)

straka_bubble_defaults = {
    'nlayers': 32,
    'dt': 1.0,
    'tmax': 900.,
    'dumpfreq': 225,
    'dirname': 'straka_bubble'
}


def straka_bubble(
        nlayers=straka_bubble_defaults['nlayers'],
        dt=straka_bubble_defaults['dt'],
        tmax=straka_bubble_defaults['tmax'],
        dumpfreq=straka_bubble_defaults['dumpfreq'],
        dirname=straka_bubble_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    domain_width = 51200.     # domain width (m)
    domain_height = 6400.     # domain height (m)
    zc = 3000.                # vertical centre of perturbation (m)
    xr = 4000.                # horizontal radius of perturbation (m)
    zr = 2000.                # vertical radius of perturbation (m)
    T_pert = -7.5             # strength of temperature perturbation (K)
    Tsurf = 300.0             # background theta value (K)
    kappa = 75.               # diffusivity parameter (m^2/s)
    mu0 = 10.                 # interior penalty parameter (1/m)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    delta = domain_height/nlayers
    ncolumns = 8 * nlayers

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=delta)

    # Equation
    parameters = CompressibleParameters(mesh)
    diffusion_params = DiffusionParameters(mesh, kappa=kappa, mu=mu0/delta)
    diffused_fields = [("u", diffusion_params), ("theta", diffusion_params)]
    eqns = CompressibleEulerEquations

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=True, dump_nc=False,
        dumplist=['u']
    )
    diagnostic_fields = [
        CourantNumber(), Perturbation('theta'), Perturbation('rho')
    ]

    # Linear solver
    linear_solver = CompressibleSolver

    model = Model(mesh, dt, parameters, eqns, output,
                  linear_solver=linear_solver, supg=True,
                  diffused_fields=diffused_fields,
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

    # Isentropic background state
    theta_b = Function(Vt).interpolate(Tsurf)
    rho_b = Function(Vr)
    exner = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, exner0=exner, solve_for_rho=True
    )

    x, z = SpatialCoordinate(mesh)
    xc = 0.5*domain_width
    r = sqrt(((x - xc)/xr)**2 + ((z - zc)/zr)**2)
    T_pert_expr = conditional(
        r > 1.,
        0.,
        0.5*T_pert*(1. + cos(pi*r))
    )

    # Set initial fields
    zero = Constant(0.0, domain=mesh)
    u0.project(as_vector([zero, zero]))
    theta0.interpolate(theta_b + T_pert_expr*exner)
    rho0.assign(rho_b)

    # Reference profiles
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
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=straka_bubble_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=straka_bubble_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=straka_bubble_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=straka_bubble_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=straka_bubble_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    straka_bubble(**vars(args))
