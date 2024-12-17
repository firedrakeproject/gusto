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
import pytest
import numpy as np

from firedrake import (
    SpatialCoordinate, as_vector, pi, sqrt, min_value, exp, cos, sin, assemble, dx, inner, Function
)
from firedrake.adjoint import *
from gusto import (
    Domain, IO, OutputParameters, Timestepper, RK4, DGUpwind,
    ShallowWaterParameters, ShallowWaterEquations, lonlatr_from_xyz,
    InstantRain, SWSaturationAdjustment, WaterVapour, CloudWater,
    Rain, GeneralIcosahedralSphereMesh
)


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    from firedrake.adjoint import annotate_tape, continue_annotation
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


def test_moist_thermal_williamson_5(
        tmpdir, ncells_per_edge=8, dt=600, tmax=50.*24.*60.*60.,
        dumpfreq=2880
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #
    radius = 6371220.           # planetary radius (m)
    mean_depth = 5960           # reference depth (m)
    g = 9.80616                 # acceleration due to gravity (m/s^2)
    u_max = 20.                 # max amplitude of the zonal wind (m/s)
    epsilon = 1/300             # linear air expansion coeff (1/K)
    theta_SP = -40*epsilon      # value of theta at south pole (no units)
    theta_EQ = 30*epsilon       # value of theta at equator (no units)
    theta_NP = -20*epsilon      # value of theta at north pole (no units)
    mu1 = 0.05                  # scaling for theta with longitude (no units)
    mu2 = 0.98                  # proportion of qsat to make init qv (no units)
    q0 = 135                    # qsat scaling, gives init q_v of ~0.02, (kg/kg)
    beta2 = 10*g                # buoyancy-vaporisation factor (m/s^2)
    nu = 20.                    # qsat factor in exponent (no units)
    qprecip = 1e-4              # cloud to rain conversion threshold (kg/kg)
    gamma_r = 1e-3              # rain-coalescence implicit factor
    mountain_height = 2000.     # height of mountain (m)
    R0 = pi/9.                  # radius of mountain (rad)
    lamda_c = -pi/2.            # longitudinal centre of mountain (rad)
    phi_c = pi/6.               # latitudinal centre of mountain (rad)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1
    u_eqn_type = 'vector_invariant_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    domain = Domain(mesh, dt, "BDM", element_order)
    x, y, z = SpatialCoordinate(mesh)
    lamda, phi, _ = lonlatr_from_xyz(x, y, z)

    # Equation: coriolis
    parameters = ShallowWaterParameters(H=mean_depth, g=g)
    Omega = parameters.Omega
    fexpr = 2*Omega*z/radius

    # Equation: topography
    rsq = min_value(R0**2, (lamda - lamda_c)**2 + (phi - phi_c)**2)
    r = sqrt(rsq)
    tpexpr = mountain_height * (1 - r/R0)

    # Equation: moisture
    tracers = [
        WaterVapour(space='DG'), CloudWater(space='DG'), Rain(space='DG')
    ]
    eqns = ShallowWaterEquations(
        domain, parameters, fexpr=fexpr, bexpr=tpexpr, thermal=True,
        active_tracers=tracers, u_transport_option=u_eqn_type
    )

    # I/O
    output = OutputParameters(
        dirname=str(tmpdir), dumplist_latlon=['D'], dumpfreq=dumpfreq,
        dump_vtus=True, dump_nc=False, log_courant=False,
        dumplist=['D', 'b', 'water_vapour', 'cloud_water']
    )
    io = IO(domain, output)

    # Physics ------------------------------------------------------------------
    # Saturation function -- first define simple expression
    def q_sat(b, D):
        return (q0/(g*D + g*tpexpr)) * exp(nu*(1 - b/g))

    # Function to pass to physics (takes mixed function as argument)
    def phys_sat_func(x_in):
        D = x_in.sub(1)
        b = x_in.sub(2)
        return q_sat(b, D)

    # Feedback proportionality is dependent on D and b
    def gamma_v(x_in):
        D = x_in.sub(1)
        b = x_in.sub(2)
        return 1.0 / (1.0 + nu*beta2/g*q_sat(b, D))

    SWSaturationAdjustment(
        eqns, phys_sat_func, time_varying_saturation=True,
        parameters=parameters, thermal_feedback=True,
        beta2=beta2, gamma_v=gamma_v, time_varying_gamma_v=True
    )

    InstantRain(
        eqns, qprecip, vapour_name="cloud_water", rain_name="rain",
        gamma_r=gamma_r
    )

    transport_methods = [
        DGUpwind(eqns, field_name) for field_name in eqns.field_names
    ]

    # Timestepper
    stepper = Timestepper(
        eqns, RK4(domain), io, spatial_methods=transport_methods
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")
    v0 = stepper.fields("water_vapour")
    c0 = stepper.fields("cloud_water")
    r0 = stepper.fields("rain")

    uexpr = as_vector([-u_max*y/radius, u_max*x/radius, 0.0])

    Dexpr = (
        mean_depth - tpexpr
        - (radius * Omega * u_max + 0.5*u_max**2)*(z/radius)**2/g
    )

    # Expression for initial buoyancy - note the bracket around 1-mu
    theta_expr = (
        2/(pi**2) * (
            phi*(phi - pi/2)*theta_SP
            - 2*(phi + pi/2) * (phi - pi/2)*(1 - mu1)*theta_EQ
            + phi*(phi + pi/2)*theta_NP
        )
        + mu1*theta_EQ*cos(phi)*sin(lamda)
    )
    bexpr = g * (1 - theta_expr)

    # Expression for initial vapour depends on initial saturation
    vexpr = mu2 * q_sat(bexpr, Dexpr)

    # Initialise (cloud and rain initially zero)
    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)
    v0.interpolate(vexpr)
    c0.assign(0.0)
    r0.assign(0.0)

    # ----------------------------------------------------------------- #
    # Run
    # ----------------------------------------------------------------- #
    stepper.run(t=0, tmax=dt)

    J = assemble(0.5*inner(u0, u0)*dx + 0.5*g*D0**2*dx)

    Jhat = ReducedFunctional(J, Control(D0))

    assert np.allclose(Jhat(D0), J)

    assert taylor_test(Jhat, D0, Function(D0.function_space()).interpolate(Dexpr)) > 1.95
