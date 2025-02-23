"""
The Eady-Boussinesq problem in a vertical slice (one cell thick in the
y-direction) solved using the incompressible Boussinesq equations. The problem
is described in Yamazaki, Shipton, Cullen, Mitchell and Cotter, 2017:
``Vertical slice modelling of nonlinear Eady waves using a compatible finite
element method''.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from gusto import (
    Domain, IncompressibleEadyParameters, IncompressibleEadyEquations,
    IncompressibleGeostrophicImbalance, SawyerEliassenU, Perturbation,
    boussinesq_hydrostatic_balance, BoussinesqSolver, SemiImplicitQuasiNewton,
    OutputParameters, IO, SSPRK3, DGUpwind, SUPGOptions, YComponent
)
from firedrake import (
    as_vector, SpatialCoordinate, solve, ds_t, ds_b, PeriodicRectangleMesh,
    ExtrudedMesh, cos, sin, cosh, sinh, tanh, pi, Function, sqrt, TrialFunction,
    TestFunction, inner, dx, div, FacetNormal,
)

incompressible_eady_defaults = {
    'ncolumns': 300,
    'nlayers': 10,
    'dt': 300.0,
    'tmax': 11*24*60*60.,
    'dumpfreq': 288,
    'dirname': 'incompressible_eady'
}


def incompressible_eady(
        ncolumns=incompressible_eady_defaults['ncolumns'],
        nlayers=incompressible_eady_defaults['nlayers'],
        dt=incompressible_eady_defaults['dt'],
        tmax=incompressible_eady_defaults['tmax'],
        dumpfreq=incompressible_eady_defaults['dumpfreq'],
        dirname=incompressible_eady_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    domain_width = 2000000.      # Width of domain (m)
    domain_thickness = 100000.   # Thickness of domain (m)
    domain_height = 10000.       # Height of domain (m)
    f = 1.e-04                   # Coriolis parameter (1/s)
    a = -4.5                     # Amplitude of the perturbation (m/s)
    Bu = 0.5                     # Burger number
    N = sqrt(2.5e-5)             # Brunt-Vaisala frequency (1/s)
    g = 10.0                     # Acceleration due to gravity (m/s^2)

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain -- 2D periodic base mesh which is one cell thick
    base_mesh = PeriodicRectangleMesh(
        ncolumns, 1, domain_width, domain_thickness, quadrilateral=True)
    mesh = ExtrudedMesh(
        base_mesh, layers=nlayers, layer_height=domain_height/nlayers
    )
    domain = Domain(mesh, dt, "RTCF", 1)

    # Equation
    parameters = IncompressibleEadyParameters(
        N=N, H=domain_height, L=0.5*domain_width, f=f, fourthorder=True,
        deltax=domain_width/ncolumns, deltaz=domain_height/nlayers, Omega=0.5*f
    )
    eqns = IncompressibleEadyEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    diagnostic_fields = [
        YComponent('u'), IncompressibleGeostrophicImbalance(eqns),
        SawyerEliassenU(eqns), Perturbation('p'), Perturbation('b')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes and methods
    b_opts = SUPGOptions()
    transport_schemes = [
        SSPRK3(domain, "u"),
        SSPRK3(domain, "b", options=b_opts)
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "b", ibp=b_opts.ibp)
    ]

    # Linear solve
    linear_solver = BoussinesqSolver(eqns)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transport_schemes, transport_methods,
        linear_solver=linear_solver
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # Initial conditions
    u0 = stepper.fields("u")
    b0 = stepper.fields("b")
    p0 = stepper.fields("p")

    # spaces
    Vu = domain.spaces("HDiv")
    Vb = domain.spaces("theta")
    Vp = domain.spaces("DG")

    # parameters
    x, _, z = SpatialCoordinate(mesh)
    Nsq = parameters.Nsq

    # background buoyancy
    bref = (z - 0.5*domain_height)*Nsq
    b_b = Function(Vb).project(bref)

    # buoyancy perturbation
    coth = lambda x: cosh(x)/sinh(x)
    Z = lambda z: Bu*((z/domain_height)-0.5)

    n = Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))

    L = 0.5*domain_width
    b_exp = a*sqrt(Nsq)*(
        -(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)
        - n*Bu*cosh(Z(z))*sin(pi*(x-L)/L)
    )
    b_pert = Function(Vb).interpolate(b_exp)

    # set total buoyancy
    b0.project(b_b + b_pert)

    # calculate hydrostatic pressure
    p_b = Function(Vp)
    boussinesq_hydrostatic_balance(eqns, b_b, p_b)
    boussinesq_hydrostatic_balance(eqns, b0, p0)

    # set x component of velocity
    dbdy = parameters.dbdy
    u = -dbdy/f*(z-0.5*domain_height)

    # set y component of velocity
    v = Function(Vp).assign(0.)

    g = TrialFunction(Vu)
    wg = TestFunction(Vu)

    n = FacetNormal(mesh)

    a = inner(wg, g)*dx
    L = -div(wg)*p0*dx + inner(wg, n)*p0*(ds_t + ds_b)
    pgrad = Function(Vu)
    solve(a == L, pgrad)

    # get initial v
    Vp = p0.function_space()
    phi = TestFunction(Vp)
    m = TrialFunction(Vp)

    a = f*phi*m*dx
    L = phi*pgrad[0]*dx
    solve(a == L, v)

    # set initial u
    u_exp = as_vector([u, v, 0.])
    u0.project(u_exp)

    # set the background profiles
    stepper.set_reference_profiles([('p', p_b), ('b', b_b)])

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
        default=incompressible_eady_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=incompressible_eady_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=incompressible_eady_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=incompressible_eady_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=incompressible_eady_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=incompressible_eady_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    incompressible_eady(**vars(args))
