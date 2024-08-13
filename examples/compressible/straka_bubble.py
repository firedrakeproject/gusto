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
    Function, sqrt, conditional
)
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, SUPGOptions, CourantNumber, Perturbation,
    DiffusionParameters, InteriorPenaltyDiffusion, BackwardEuler,
    CompressibleParameters, CompressibleEulerEquations, CompressibleSolver,
    compressible_hydrostatic_balance
)

straka_bubble_defaults = {
    'ncolumns': 256,
    'nlayers': 32,
    'dt': 1.0,
    'tmax': 900.,
    'dumpfreq': 225,
    'dirname': 'straka_bubble'
}

def straka_bubble(
        ncolumns=straka_bubble_defaults['ncolumns'],
        nlayers=straka_bubble_defaults['nlayers'],
        dt=straka_bubble_defaults['dt'],
        tmax=straka_bubble_defaults['tmax'],
        dumpfreq=straka_bubble_defaults['dumpfreq'],
        dirname=straka_bubble_defaults['dirname']
):

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

if '--running-tests' in sys.argv:
    res_dt = {800.: 4.}
    tmax = 4.
    ndumps = 1
else:
    res_dt = {800.: 4., 400.: 2., 200.: 1., 100.: 0.5, 50.: 0.25}
    tmax = 15.*60.
    ndumps = 4


L = 51200.

# build volume mesh
H = 6400.  # Height position of the model top

for delta, dt in res_dt.items():

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    nlayers = int(H/delta)  # horizontal layers
    columns = int(L/delta)  # number of columns
    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    parameters = CompressibleParameters()
    u_diffusion_opts = DiffusionParameters(kappa=75., mu=10./delta)
    theta_diffusion_opts = DiffusionParameters(kappa=75., mu=10./delta)
    diffusion_options = [("u", u_diffusion_opts), ("theta", theta_diffusion_opts)]
    eqns = CompressibleEulerEquations(domain, parameters,
                                      diffusion_options=diffusion_options)

    # I/O
    dirname = "straka_dx%s_dt%s" % (delta, dt)
    dumpfreq = int(tmax / (ndumps*dt))
    output = OutputParameters(
        dirname=dirname,
        dumpfreq=dumpfreq,
        dumplist=['u'],
    )
    diagnostic_fields = [CourantNumber(), Perturbation('theta'), Perturbation('rho')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    theta_opts = SUPGOptions()
    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "rho"),
                          SSPRK3(domain, "theta", options=theta_opts)]
    transport_methods = [DGUpwind(eqns, "u"),
                         DGUpwind(eqns, "rho"),
                         DGUpwind(eqns, "theta", ibp=theta_opts.ibp)]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Diffusion schemes
    diffusion_schemes = [BackwardEuler(domain, "u"),
                         BackwardEuler(domain, "theta")]
    diffusion_methods = [InteriorPenaltyDiffusion(eqns, "u", u_diffusion_opts),
                         InteriorPenaltyDiffusion(eqns, "theta", theta_diffusion_opts)]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                      spatial_methods=transport_methods+diffusion_methods,
                                      linear_solver=linear_solver,
                                      diffusion_schemes=diffusion_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # spaces
    Vu = domain.spaces("HDiv")
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    # Isentropic background state
    Tsurf = Constant(300.)

    theta_b = Function(Vt).interpolate(Tsurf)
    rho_b = Function(Vr)
    exner = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta_b, rho_b, exner0=exner,
                                     solve_for_rho=True)

    x = SpatialCoordinate(mesh)
    a = 5.0e3
    xc = 0.5*L
    xr = 4000.
    zc = 3000.
    zr = 2000.
    r = sqrt(((x[0]-xc)/xr)**2 + ((x[1]-zc)/zr)**2)
    T_pert = conditional(r > 1., 0., -7.5*(1.+cos(pi*r)))
    theta0.interpolate(theta_b + T_pert*exner)
    rho0.assign(rho_b)

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
        default=straka_bubble_defaults['ncolumns']
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
