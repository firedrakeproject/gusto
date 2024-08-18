"""
This example uses the hydrostatic compressible Euler equations to solve the
vertical slice gravity wave test case of Skamarock and Klemp, 1994:
``Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique'',
MWR.

Potential temperature is transported using SUPG, and the degree 1 elements are
used. This also uses a mesh which is one cell thick in the y-direction.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    as_vector, SpatialCoordinate, PeriodicRectangleMesh, ExtrudedMesh, exp, sin,
    Function, pi
)
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, SUPGOptions, CourantNumber, Perturbation,
    CompressibleParameters, HydrostaticCompressibleEulerEquations,
    CompressibleSolver, compressible_hydrostatic_balance
)

skamarock_klemp_hydrostatic_defaults = {
    'ncolumns': 150,
    'nlayers': 10,
    'dt': 25.0,
    'tmax': 60000.,
    'dumpfreq': 1200,
    'dirname': 'skamarock_klemp_hydrostatic'
}


def skamarock_klemp_hydrostatic(
        ncolumns=skamarock_klemp_hydrostatic_defaults['ncolumns'],
        nlayers=skamarock_klemp_hydrostatic_defaults['nlayers'],
        dt=skamarock_klemp_hydrostatic_defaults['dt'],
        tmax=skamarock_klemp_hydrostatic_defaults['tmax'],
        dumpfreq=skamarock_klemp_hydrostatic_defaults['dumpfreq'],
        dirname=skamarock_klemp_hydrostatic_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    domain_width = 6.0e6               # Width of domain in x direction (m)
    domain_length = 1.0e4              # Length of domain in y direction (m)
    domain_height = 1.0e4              # Height of domain (m)
    Tsurf = 300.                       # Temperature at surface (K)
    wind_initial = 20.                 # Initial wind in x direction (m/s)
    pert_width = 5.0e3                 # Width parameter of perturbation (m)
    deltaTheta = 1.0e-2                # Magnitude of theta perturbation (K)
    N = 0.01                           # Brunt-Vaisala frequency (1/s)
    Omega = 0.5e-4                     # Planetary rotation rate (1/s)
    pressure_gradient_y = -1.0e-4*20   # Prescribed force in y direction (m/s^2)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain -- 3D volume mesh
    base_mesh = PeriodicRectangleMesh(
        ncolumns, 1, domain_width, domain_length, quadrilateral=True
    )
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    domain = Domain(mesh, dt, "RTCF", element_order)

    # Equation
    parameters = CompressibleParameters()
    balanced_pg = as_vector((0., pressure_gradient_y, 0.))
    eqns = HydrostaticCompressibleEulerEquations(
        domain, parameters, extra_terms=[("u", balanced_pg)]
    )

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=True, dump_nc=False,
        dumplist=['u'],
    )
    diagnostic_fields = [CourantNumber(), Perturbation('theta'), Perturbation('rho')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    theta_opts = SUPGOptions()
    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "rho"),
        SSPRK3(domain, "theta", options=theta_opts)
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho"),
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

    x, _, z = SpatialCoordinate(mesh)

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    theta_pert = (
        deltaTheta * sin(pi*z/domain_height)
        / (1 + (x - domain_width/2)**2 / pert_width**2)
    )
    theta0.interpolate(theta_b + theta_pert)

    compressible_hydrostatic_balance(eqns, theta_b, rho_b, solve_for_rho=True)

    rho0.assign(rho_b)
    u0.project(as_vector([wind_initial, 0.0, 0.0]))

    stepper.set_reference_profiles([('rho', rho_b),
                                    ('theta', theta_b)])

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
        default=skamarock_klemp_hydrostatic_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=skamarock_klemp_hydrostatic_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=skamarock_klemp_hydrostatic_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=skamarock_klemp_hydrostatic_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=skamarock_klemp_hydrostatic_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=skamarock_klemp_hydrostatic_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    skamarock_klemp_hydrostatic(**vars(args))
