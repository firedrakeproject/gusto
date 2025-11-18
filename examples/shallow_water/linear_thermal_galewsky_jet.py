"""
A linearised form of the steady thermal Galewsky jet. The initial conditions are
taken from Hartney et al, 2024: ``A compatible finite element discretisation
for moist shallow water equations'' (without the perturbation).

This uses an icosahedral mesh of the sphere.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    SpatialCoordinate, pi, assemble, dx, Constant, exp, conditional, Function,
    cos
)
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, DefaultTransport,
    DGUpwind, ForwardEuler, ShallowWaterParameters, NumericalIntegral,
    LinearThermalShallowWaterEquations, GeneralIcosahedralSphereMesh,
    ZonalComponent, ThermalSWSolver, lonlatr_from_xyz, xyz_vector_from_lonlatr,
    RelativeVorticity, MeridionalComponent
)

import numpy as np

linear_thermal_galewsky_jet_defaults = {
    'ncells_per_edge': 12,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': 6.*24.*60.*60.,    # 6 days
    'dumpfreq': 96,            # once per day with default options
    'dirname': 'linear_thermal_galewsky'
}


def linear_thermal_galewsky_jet(
        ncells_per_edge=linear_thermal_galewsky_jet_defaults['ncells_per_edge'],
        dt=linear_thermal_galewsky_jet_defaults['dt'],
        tmax=linear_thermal_galewsky_jet_defaults['tmax'],
        dumpfreq=linear_thermal_galewsky_jet_defaults['dumpfreq'],
        dirname=linear_thermal_galewsky_jet_defaults['dirname']
):
    # ----------------------------------------------------------------- #
    # Parameters for test case
    # ----------------------------------------------------------------- #

    R = 6371220.         # planetary radius (m)
    H = 10000.           # reference depth (m)
    u_max = 80.0         # Max amplitude of the zonal wind (m/s)
    phi0 = pi/7.         # latitude of the southern boundary of the jet (rad)
    phi1 = pi/2. - phi0  # latitude of the northern boundary of the jet (rad)
    db = 1.0             # diff in buoyancy between equator and poles (m/s^2)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ----------------------------------------------------------------- #
    # Set up model objects
    # ----------------------------------------------------------------- #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(R, ncells_per_edge, degree=2)
    xyz = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', element_order)

    # Equation
    parameters = ShallowWaterParameters(mesh, H=H)
    eqns = LinearThermalShallowWaterEquations(domain, parameters)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False,
        dumplist=['D', 'b']
    )
    diagnostic_fields = [
        ZonalComponent('u'), MeridionalComponent('u'), RelativeVorticity()
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transport_schemes = [ForwardEuler(domain, "D")]
    transport_methods = [DefaultTransport(eqns, "D"), DGUpwind(eqns, "b")]

    # Linear solver
    linear_solver = ThermalSWSolver(eqns)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transport_schemes, transport_methods,
        linear_solver=linear_solver
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0_field = stepper.fields("u")
    D0_field = stepper.fields("D")
    b0_field = stepper.fields("b")

    # Parameters
    g = parameters.g
    Omega = parameters.Omega
    e_n = np.exp(-4./((phi1-phi0)**2))

    _, lat, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
    lat_VD = Function(D0_field.function_space()).interpolate(lat)

    # ------------------------------------------------------------------------ #
    # Obtain u and D (by integration of analytic expression)
    # ------------------------------------------------------------------------ #

    # Buoyancy
    bexpr = g - db*cos(lat)

    # Wind -- UFL expression
    u_zonal = conditional(
        lat <= phi0, 0.0,
        conditional(
            lat >= phi1, 0.0,
            u_max / e_n * exp(1.0 / ((lat - phi0) * (lat - phi1)))
        )
    )
    uexpr = xyz_vector_from_lonlatr(u_zonal, Constant(0.0), Constant(0.0), xyz)

    # Numpy function
    def u_func(y):
        u_array = np.where(
            y <= phi0, 0.0,
            np.where(
                y >= phi1, 0.0,
                u_max / e_n * np.exp(1.0 / ((y - phi0) * (y - phi1)))
            )
        )
        return u_array

    # Function for depth field in terms of u function
    def h_func(y):
        h_array = u_func(y)*float(R)/float(g)*(
            2*float(Omega)*np.sin(y)
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
    b0_field.interpolate(bexpr)

    # Set reference profiles
    Dbar = Function(D0_field.function_space()).assign(H)
    bbar = Function(b0_field.function_space()).interpolate(g)
    stepper.set_reference_profiles([('D', Dbar), ('b', bbar)])

    # ----------------------------------------------------------------- #
    # Run
    # ----------------------------------------------------------------- #

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
        help="The number of cells per edge of icosahedron",
        type=int,
        default=linear_thermal_galewsky_jet_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=linear_thermal_galewsky_jet_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=linear_thermal_galewsky_jet_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=linear_thermal_galewsky_jet_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=linear_thermal_galewsky_jet_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    linear_thermal_galewsky_jet(**vars(args))
