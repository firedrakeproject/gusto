"""
Tests discretisations of the forced advection equation. This test describes
transport of water vapour, which is converted to rain where it exceeds a
prescribed saturation profile. The initial condition and saturation profile are
chosen to give an analytic solution. The test compares the errornorm for the
resulting field against the analytic solution to check that they agree, within
a specified tolerance.
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, SpatialCoordinate,
                       VectorFunctionSpace, conditional, acos, cos, pi, sin,
                       as_vector, errornorm)


def run_forced_advection(tmpdir):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    Lx = 100
    delta_x = 2.0
    nx = int(Lx/delta_x)
    mesh = PeriodicIntervalMesh(nx, Lx)
    x = SpatialCoordinate(mesh)[0]

    dt = 0.2
    domain = Domain(mesh, dt, "CG", 1)

    VD = domain.spaces("DG")
    Vu = VectorFunctionSpace(mesh, "CG", 1)

    # Equation
    u_max = 1
    C0 = 0.6
    K0 = 0.3
    Csat = 0.75
    Ksat = 0.25
    tmax = 85

    # saturation profile
    msat_expr = Csat + (Ksat * cos(2*pi*(x/Lx)))
    msat = Function(VD)
    msat.interpolate(msat_expr)

    # Rain is a first tracer
    rain = Rain(space='DG',
                transport_eqn=TransportEquationType.no_transport)

    # Also, have water_vapour as a tracer:
    water_vapour = WaterVapour(space='DG')

    meqn = CoupledTransportEquation(domain, active_tracers=[rain, water_vapour], Vu=Vu)

    transport_method = DGUpwind(meqn, "water_vapour")
    physics_parametrisations = [InstantRain(meqn, msat, rain_name="rain",
                                            parameters=None)]

    # I/O
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=1)
    diagnostic_fields = [CourantNumber()]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Time Stepper
    stepper = PrescribedTransport(meqn, RK4(domain), io, transport_method,
                                  physics_parametrisations=physics_parametrisations)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # initial moisture profile
    mexpr = C0 + K0*cos((2*pi*x)/Lx)

    stepper.fields("u").project(as_vector([u_max]))
    stepper.fields("water_vapour").project(mexpr)

    # Start with no rain:
    no_rain = 0*x
    stepper.fields("rain").interpolate(no_rain)

    # exact rainfall profile (analytically)
    r_exact = stepper.fields("r_exact", space=VD)
    lim1 = Lx/(2*pi) * acos((C0 + K0 - Csat)/Ksat)
    lim2 = Lx/2
    coord = (Ksat*cos(2*pi*x/Lx) + Csat - C0)/K0
    exact_expr = 2*Ksat*sin(2*pi*x/Lx)*acos(coord)
    r_expr = conditional(x < lim2, conditional(x > lim1, exact_expr, 0), 0)
    r_exact.interpolate(r_expr)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(0, tmax=tmax)

    error = errornorm(r_exact, stepper.fields("rain"))

    return error


def test_forced_advection(tmpdir):

    tol = 0.1
    error = run_forced_advection(tmpdir)
    assert error < tol, 'The error in the forced advection equation is greater than the permitted tolerance'
