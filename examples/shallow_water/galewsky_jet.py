"""
The unsteady jet test case on the sphere of Galewsky, Scott & Polvani, 2004:
``An initial-value problem for testing numerical models of the global
shallow-water equations'', Tellus A (DMO).

The test adds a perturbation to an unsteady mid-latitude jet, which gradually
unfurls.

The setup implemented here uses the cubed sphere with the degree 1 spaces.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    SpatialCoordinate, pi, conditional, exp, cos, assemble, dx, Constant,
    Function, sqrt
)
from gusto import (
    Domain, IO, OutputParameters, GeneralCubedSphereMesh, RelativeVorticity,
    lonlatr_from_xyz, xyz_vector_from_lonlatr, NumericalIntegral,
    ShallowWaterEquations, ShallowWaterParameters, SSPRK3, DGUpwind,
    TRBDF2QuasiNewton, ZonalComponent, MeridionalComponent,
    LinearTimesteppingSolver
)
import numpy as np

galewsky_jet_defaults = {
    'ncells_per_edge': 24,     # number of cells per cubed sphere panel edge
    'dt': 1200.0,              # 20 minutes
    'tmax': 6.*24.*60.*60.,    # 6 days
    'dumpfreq': 72,            # once per day with default options
    'dirname': 'galewsky_jet'
}


def galewsky_jet(
        ncells_per_edge=galewsky_jet_defaults['ncells_per_edge'],
        dt=galewsky_jet_defaults['dt'],
        tmax=galewsky_jet_defaults['tmax'],
        dumpfreq=galewsky_jet_defaults['dumpfreq'],
        dirname=galewsky_jet_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    radius = 6371220.    # radius of the planet, in m
    H = 10000.           # mean (and reference) depth, in m
    umax = 80.0          # amplitude of jet wind speed, in m/s
    phi0 = pi/7          # lower latitude of initial jet, in rad
    phi1 = pi/2 - phi0   # upper latitude of initial jet, in rad
    phi2 = pi/4          # central latitude of perturbation to jet, in rad
    alpha = 1.0/3        # zonal width parameter of perturbation, in rad
    beta = 1.0/15        # meridional width parameter of perturbation, in rad
    h_hat = 120.0        # strength of perturbation, in m

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    degree = 1
    hdiv_family = 'RTCF'
    u_eqn_type = 'vector_advection_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralCubedSphereMesh(radius, ncells_per_edge, degree=2)
    domain = Domain(mesh, dt, hdiv_family, degree)

    # Equation
    xyz = SpatialCoordinate(mesh)
    parameters = ShallowWaterParameters(mesh, H=H)
    Omega = parameters.Omega
    fexpr = 2*Omega*xyz[2]/radius
    eqns = ShallowWaterEquations(
        domain, parameters, fexpr=fexpr, u_transport_option=u_eqn_type
    )

    # I/O and diagnostics
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dumplist=['D']
    )
    diagnostic_fields = [
        RelativeVorticity(), ZonalComponent('u'), MeridionalComponent('u')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transported_fields = [SSPRK3(domain, "u"), SSPRK3(domain, "D")]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

    gamma = 1 - sqrt(2)/2
    gamma2 = (1 - 2*gamma)/(2 - 2*gamma)
    tr_solver = LinearTimesteppingSolver(eqns, alpha=gamma)
    bdf_solver = LinearTimesteppingSolver(eqns, alpha=gamma2)

    # Time stepper
    stepper = TRBDF2QuasiNewton(
        eqns, io, transported_fields, transport_methods,
        tr_solver=tr_solver, bdf_solver=bdf_solver, gamma=gamma
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0_field = stepper.fields("u")
    D0_field = stepper.fields("D")

    # Parameters
    g = parameters.g
    Omega = parameters.Omega
    e_n = np.exp(-4./((phi1-phi0)**2))

    lon, lat, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
    lat_VD = Function(D0_field.function_space()).interpolate(lat)

    # ------------------------------------------------------------------------ #
    # Obtain u and D (by integration of analytic expression)
    # ------------------------------------------------------------------------ #

    # Wind -- UFL expression
    u_zonal = conditional(
        lat <= phi0, 0.0,
        conditional(
            lat >= phi1, 0.0,
            umax / e_n * exp(1.0 / ((lat - phi0) * (lat - phi1)))
        )
    )
    uexpr = xyz_vector_from_lonlatr(u_zonal, Constant(0.0), Constant(0.0), xyz)

    # Numpy function
    def u_func(y):
        u_array = np.where(
            y <= phi0, 0.0,
            np.where(
                y >= phi1, 0.0,
                umax / e_n * np.exp(1.0 / ((y - phi0) * (y - phi1)))
            )
        )
        return u_array

    # Function for depth field in terms of u function
    def h_func(y):
        h_array = u_func(y)*radius/g*(
            2*Omega*np.sin(y)
            + u_func(y)*np.tan(y)/radius
        )

        return h_array

    # Find h from numerical integral
    D0_integral = Function(D0_field.function_space())
    h_integral = NumericalIntegral(-pi/2, pi/2)
    h_integral.tabulate(h_func)
    D0_integral.dat.data[:] = h_integral.evaluate_at(lat_VD.dat.data[:])
    Dexpr = H - D0_integral

    # Obtain fields
    u0_field.project(uexpr)
    D0_field.interpolate(Dexpr)

    # Adjust mean value of initial D
    C = Function(D0_field.function_space()).assign(Constant(1.0))
    area = assemble(C*dx)
    Dmean = assemble(D0_field*dx)/area
    D0_field -= Dmean
    D0_field += Constant(H)

    # Background field, store in object for use in diagnostics
    Dbar = Function(D0_field.function_space()).assign(D0_field)

    # ------------------------------------------------------------------------ #
    # Apply perturbation
    # ------------------------------------------------------------------------ #

    h_pert = h_hat*cos(lat)*exp(-(lon/alpha)**2)*exp(-((phi2-lat)/beta)**2)
    D0_field.interpolate(Dexpr + h_pert)

    stepper.set_reference_profiles([('D', Dbar)])

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
        '--ncells_per_edge',
        help="The number of cells per edge of cubed sphere panel",
        type=int,
        default=galewsky_jet_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=galewsky_jet_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=galewsky_jet_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=galewsky_jet_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=galewsky_jet_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    galewsky_jet(**vars(args))
