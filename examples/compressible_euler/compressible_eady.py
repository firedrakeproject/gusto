"""
This solves the Eady problem in a vertical slice (one cell thick in the
y-direction) using the compressible Euler equations, following
Yamazaki and Cotter, 2025:
``A vertical slice frontogenesis test case for compressible nonhydrostatic
dynamical cores of atmospheric models''.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from gusto import (
    Domain, CompressibleEadyParameters, CompressibleEadyEquations,
    Perturbation, compressible_hydrostatic_balance,
    CompressibleSolver, SemiImplicitQuasiNewton, OutputParameters, IO,
    SSPRK3, DGUpwind, SUPGOptions, YComponent, Exner
)
from gusto import thermodynamics as tde
from firedrake import (
    as_vector, SpatialCoordinate, solve, ds_b, ds_t, PeriodicRectangleMesh,
    ExtrudedMesh, assemble, exp, cos, sin, cosh, sinh, tanh, pi, Function, sqrt,
    TrialFunction, TestFunction, inner, dx, div, FacetNormal, FunctionSpace
)

compressible_eady_defaults = {
    'ncolumns': 300,
    'nlayers': 10,
    'dt': 300.0,
    'tmax': 11*24*60*60.,
    'dumpfreq': 288,
    'dirname': 'compressible_eady'
}


def compressible_eady(
        ncolumns=compressible_eady_defaults['ncolumns'],
        nlayers=compressible_eady_defaults['nlayers'],
        dt=compressible_eady_defaults['dt'],
        tmax=compressible_eady_defaults['tmax'],
        dumpfreq=compressible_eady_defaults['dumpfreq'],
        dirname=compressible_eady_defaults['dirname']
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
    Pi0 = 0.864                  # Exner pressure at the surface

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
    parameters = CompressibleEadyParameters(
        N=N, H=domain_height, f=f, Pi0=Pi0, g=g, Omega=0.5*f
    )
    eqns = CompressibleEadyEquations(
        domain, parameters, u_transport_option='vector_advection_form'
    )

    # I/O
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    diagnostic_fields = [
        YComponent('u'), Exner(parameters), Perturbation('theta'),
        Perturbation('Exner')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes and methods
    theta_opts = SUPGOptions()
    transport_schemes = [
        SSPRK3(domain, "u"),
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
        eqns, io, transport_schemes, transport_methods,
        linear_solver=linear_solver
    )

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

    # first setup the background buoyancy profile
    # z.grad(bref) = N**2
    x, _, z = SpatialCoordinate(mesh)
    g = parameters.g
    Nsq = parameters.Nsq
    theta_surf = parameters.theta_surf

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    theta_ref = theta_surf*exp(Nsq*(z - 0.5*domain_height)/g)
    theta_b = Function(Vt).interpolate(theta_ref)

    # set theta_pert
    coth = lambda x: cosh(x)/sinh(x)
    Z = lambda z: Bu*((z/domain_height)-0.5)

    n = Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))

    L = 0.5*domain_width
    theta_exp = a*theta_surf/g*sqrt(Nsq)*(
        -(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)
        - n*Bu*cosh(Z(z))*sin(pi*(x-L)/L)
    )
    theta_pert = Function(Vt).interpolate(theta_exp)

    # set theta0
    theta0.interpolate(theta_b + theta_pert)

    # calculate hydrostatic Pi
    rho_b = Function(Vr)
    compressible_hydrostatic_balance(eqns, theta_b, rho_b, solve_for_rho=True)
    compressible_hydrostatic_balance(eqns, theta0, rho0, solve_for_rho=True)

    Pi = tde.exner_pressure(parameters, rho0, theta0)

    # set x component of velocity
    cp = parameters.cp
    dthetady = parameters.dthetady
    u = cp*dthetady/f*(Pi-Pi0)

    # set y component of velocity by solving a problem
    v = Function(Vr).assign(0.)

    # get Pi gradient
    g = TrialFunction(Vu)
    wg = TestFunction(Vu)

    n = FacetNormal(mesh)

    a = inner(wg, g)*dx
    L = -div(wg)*Pi*dx + inner(wg, n)*Pi*(ds_t + ds_b)
    pgrad = Function(Vu)
    solve(a == L, pgrad)

    # get initial v
    m = TrialFunction(Vr)
    phi = TestFunction(Vr)

    a = phi*f*m*dx
    L = phi*cp*theta0*pgrad[0]*dx
    solve(a == L, v)

    # set initial u
    u_exp = as_vector([u, v, 0.])
    u0.project(u_exp)

    # set the background profiles
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
        default=compressible_eady_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=compressible_eady_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=compressible_eady_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=compressible_eady_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=compressible_eady_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=compressible_eady_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    compressible_eady(**vars(args))
