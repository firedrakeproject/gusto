"""
The moist thermal form of Test Case 5 (flow over a mountain) of Williamson et
al, 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

The initial conditions are taken from Zerroukat & Allen, 2015:
``A moist Boussinesq shallow water equations set for testing atmospheric
models'', JCP.

Three moist variables (vapour, cloud liquid and rain) are used. This set of
equations involves an active buoyancy field.

The example here uses the icosahedral sphere mesh and degree 1 spaces. An
explicit RK4 timestepper is used.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, exp, cos, sin)
import sys

moist_thermal_williamson_2_defaults = {
    'ncells_per_edge': 48,     # number of cells per icosahedron edge
    'dt': 300.0,               # 5 minutes
    'tmax': 6.*24.*60.*60.,    # 6 days
    'dumpfreq': 288,           # once per day with default options
    'dirname': 'moist_thermal_williamson_2'
}

def moist_thermal_williamson_2(
        ncells_per_edge=moist_thermal_williamson_2_defaults['ncells_per_edge'],
        dt=moist_thermal_williamson_2_defaults['dt'],
        tmax=moist_thermal_williamson_2_defaults['tmax'],
        dumpfreq=moist_thermal_williamson_2_defaults['dumpfreq'],
        dirname=moist_thermal_williamson_2_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    dt = 300

    if '--running-tests' in sys.argv:
        tmax = dt
        dumpfreq = 1
    else:
        day = 24*60*60
        tmax = 50*day
        ndumps = 50
        dumpfreq = int(tmax / (ndumps*dt))

    R = 6371220.
    H = 5960.
    u_max = 20.
    # moist shallow water parameters
    epsilon = 1/300
    SP = -40*epsilon
    EQ = 30*epsilon
    NP = -20*epsilon
    mu1 = 0.05
    mu2 = 0.98
    q0 = 135  # chosen to give an initial max vapour of approx 0.02
    beta2 = 10
    qprecip = 1e-4
    gamma_r = 1e-3
    # topography parameters
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    phi_c = pi/6.

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = IcosahedralSphereMesh(radius=R,
                                refinement_level=4, degree=1)
    degree = 1
    domain = Domain(mesh, dt, "BDM", degree)
    x = SpatialCoordinate(mesh)

    # Equation
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R

    # Topography
    lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])
    lsq = (lamda - lamda_c)**2
    thsq = (phi - phi_c)**2
    rsq = min_value(R0sq, lsq+thsq)
    r = sqrt(rsq)
    tpexpr = 2000 * (1 - r/R0)

    tracers = [WaterVapour(space='DG'), CloudWater(space='DG'), Rain(space='DG')]
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=tpexpr,
                                thermal=True,
                                active_tracers=tracers)

    # I/O
    dirname = "moist_thermal_williamson5"
    output = OutputParameters(
        dirname=dirname,
        dumplist_latlon=['D'],
        dumpfreq=dumpfreq,
    )
    diagnostic_fields = [Sum('D', 'topography'), CourantNumber()]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)


    # Saturation function
    def sat_func(x_in):
        h = x_in.split()[1]
        b = x_in.split()[2]
        return (q0/(g*h + g*tpexpr)) * exp(20*(1 - b/g))


    # Feedback proportionality is dependent on h and b
    def gamma_v(x_in):
        h = x_in.split()[1]
        b = x_in.split()[2]
        return (1 + beta2*(20*q0/(g*h + g*tpexpr) * exp(20*(1 - b/g))))**(-1)


    SWSaturationAdjustment(eqns, sat_func, time_varying_saturation=True,
                        parameters=parameters, thermal_feedback=True,
                        beta2=beta2, gamma_v=gamma_v,
                        time_varying_gamma_v=True)

    InstantRain(eqns, qprecip, vapour_name="cloud_water", rain_name="rain",
                gamma_r=gamma_r)

    transport_methods = [DGUpwind(eqns, field_name) for field_name in eqns.field_names]

    # Timestepper
    stepper = Timestepper(eqns, RK4(domain), io, spatial_methods=transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")
    v0 = stepper.fields("water_vapour")
    c0 = stepper.fields("cloud_water")
    r0 = stepper.fields("rain")

    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])

    g = parameters.g
    Rsq = R**2
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - tpexpr

    # Expression for initial buoyancy - note the bracket around 1-mu
    F = (2/(pi**2))*(phi*(phi-pi/2)*SP - 2*(phi+pi/2)*(phi-pi/2)*(1-mu1)*EQ + phi*(phi+pi/2)*NP)
    theta_expr = F + mu1*EQ*cos(phi)*sin(lamda)
    bexpr = g * (1 - theta_expr)

    # Expression for initial vapour depends on initial saturation
    initial_msat = q0/(g*D0 + g*tpexpr) * exp(20*theta_expr)
    vexpr = mu2 * initial_msat

    # Initialise (cloud and rain initially zero)
    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)
    v0.interpolate(vexpr)

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
        default=moist_thermal_williamson_2_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=moist_thermal_williamson_2_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=moist_thermal_williamson_2_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=moist_thermal_williamson_2_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=moist_thermal_williamson_2_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    moist_thermal_williamson_2(**vars(args))
