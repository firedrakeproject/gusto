"""
A linearised form of the steady jet of Galewsky et al 2004:
``An initial-value problem for testing numerical models of the global
shallow-water equations'', Tellus A: Dynamic Meteorology and Oceanography.

This uses an icosahedral mesh of the sphere, and the linear shallow water
equations.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (SpatialCoordinate, pi, assemble,
                       dx, Constant, ge, le, exp, cos, conditional,
                       interpolate, Function, VectorFunctionSpace,)
from gusto import (Domain, IO, OutputParameters, SemiImplicitQuasiNewton,
                   DefaultTransport, DGUpwind, ForwardEuler, SteadyStateError,
                   ShallowWaterParameters, LinearThermalShallowWaterEquations,
                   GeneralIcosahedralSphereMesh, ZonalComponent,
                   MeridionalComponent, RelativeVorticity, PotentialVorticity,
                   ThermalSWSolver, lonlatr_from_xyz, xyz_vector_from_lonlatr
                   )

import numpy as np

linear_thermal_galewsky_jet_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': 6.*24.*60.*60.,    # 5 days
    'dumpfreq': 96,            # once per day with default options
    'dirname': 'linear_thermal_galewsky_jet'
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

    R = 6371220.  # planetary radius (m)
    H = 10000.  # reference depth (m)
    u_max = 80.0  # Max amplitude of the zonal wind (m/s)
    phi0 = pi/7.  # latitude of the southern boundary of the jet (radians)
    phi1 = pi/2. - phi0  # latitude of the northern boundary of the jet (radians)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ----------------------------------------------------------------- #
    # Set up model objects
    # ----------------------------------------------------------------- #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(R, ncells_per_edge, degree=2)
    x = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', element_order)

    # Equation
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    eqns = LinearThermalShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=False, dump_vtus=True
    )
    diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D'),
                         SteadyStateError('b'), ZonalComponent('u'),
                         MeridionalComponent('u'), RelativeVorticity(),
                         PotentialVorticity()]
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

    g = Constant(parameters.g)

    D0 = stepper.fields("D")
    b0 = stepper.fields("b")

    # get lat lon coordinates
    lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])

    # expressions for meridional and zonal velocity
    en = np.exp(-4./((phi1-phi0)**2))
    u_zonal_expr = (u_max/en)*exp(1/((phi - phi0)*(phi - phi1)))
    u_zonal = conditional(ge(phi, phi0), conditional(le(phi, phi1), u_zonal_expr, 0.), 0.)
    u_merid = 0.0

    # get cartesian components of velocity
    uexpr = xyz_vector_from_lonlatr(u_zonal, u_merid, 0, x)

    # expression for buoyancy
    bexpr = g - cos(phi)
    b0.interpolate(bexpr)

    # Compute balanced initial depth

    def D_integrand(th):
        # Initial D field is calculated by integrating D_integrand w.r.t. phi
        # Assumes the input is between phi0 and phi1.
        # Note that this function operates on vectorized input.
        from numpy import exp, sin
        f = 2.0*parameters.Omega*sin(th)
        u_zon = (80.0/en)*exp(1.0/((th - phi0)*(th - phi1)))
        return u_zon*f

    def Dval(X):
        # Function to return value of D at X
        from scipy import integrate

        # Preallocate output array
        val = np.zeros(len(X))

        angles = np.zeros(len(X))

        # Minimize work by only calculating integrals for points with
        # phi between phi_0 and phi_1.
        # For phi <= phi_0, the integral is 0
        # For phi >= phi_1, the integral is constant.

        # Precalculate this constant:
        poledepth, _ = integrate.fixed_quad(D_integrand, phi0, phi1, n=64)
        poledepth *= -R/parameters.g

        angles[:] = np.arcsin(X[:, 2]/R)

        for ii in range(len(X)):
            if angles[ii] <= phi0:
                val[ii] = 0.0
            elif angles[ii] >= phi1:
                val[ii] = poledepth
            else:
                # Fixed quadrature with 64 points gives absolute errors below 1e-13
                # for a quantity of order 1e-3.
                v, _ = integrate.fixed_quad(D_integrand, phi0, angles[ii], n=64)
                val[ii] = -(R/parameters.g)*v

        return val

    def initialise_fn():
        u0 = stepper.fields("u")
        D0 = stepper.fields("D")

        u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})

        # Get coordinates to pass to Dval function
        W = VectorFunctionSpace(mesh, D0.ufl_element())

        X = interpolate(mesh.coordinates, W)
        D0.dat.data[:] = Dval(X.dat.data_ro)
        D0.interpolate(D0 - (H/(2*g) * b0))

        # Adjust mean value of initial D
        C = Function(D0.function_space()).assign(Constant(1.0))
        area = assemble(C*dx)
        Dmean = assemble(D0*dx)/area
        D0 -= Dmean
        D0 += Constant(parameters.H)

    initialise_fn()

    # Set reference profiles
    Dbar = Function(D0.function_space()).assign(H)
    bbar = Function(b0.function_space()).interpolate(g)
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
