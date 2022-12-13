"""
Tests discretisations of the forced advection equation. This test describes
transport of water vapour, which is converted to rain where it exceeds a
prescribed saturation profile. The initial condition and saturation profile are
chosen to give an analytic solution. The test compares the errornorm for the
resulting field against the analytic solution to check that they agree, within
a specified tolerance.
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, SpatialCoordinate, FunctionSpace,
                       VectorFunctionSpace, conditional, acos, cos, pi,
                       as_vector, errornorm)


def run_forced_advection(tmpdir):

    Lx = 100
    delta_x = 2.0
    nx = int(Lx/delta_x)
    mesh = PeriodicIntervalMesh(nx, Lx)
    x = SpatialCoordinate(mesh)[0]

    dt = 0.2
    domain = Domain(mesh, dt, "CG", 1)

    VD = FunctionSpace(mesh, "DG", 1)
    Vu = VectorFunctionSpace(mesh, "CG", 1)

    # set up parameters and initial conditions
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

    # initial moisture profile
    mexpr = C0 + K0*cos((2*pi*x)/Lx)

    rain = Rain(space='tracer',
                transport_eqn=TransportEquationType.no_transport)
    meqn = ForcedAdvectionEquation(domain, VD, field_name="water_vapour", Vu=Vu,
                                   active_tracers=[rain])
    physics_schemes = [(InstantRain(meqn, msat, rain_name="rain",
                                    set_tau_to_dt=True), ForwardEuler(domain))]

    output = OutputParameters(dirname=str(tmpdir), dumpfreq=1)
    diagnostic_fields = [CourantNumber()]
    io = IO(domain, meqn, output=output, diagnostic_fields=diagnostic_fields)


    meqn.fields("u").project(as_vector([u_max]))
    qv = meqn.fields("water_vapour")
    qv.project(mexpr)

    # exact rainfall profile (analytically)
    r_exact = meqn.fields("r_exact", VD)
    lim1 = Lx/(2*pi) * acos((C0 + K0 - Csat)/Ksat)
    lim2 = Lx/2
    coord = (Ksat*cos(2*pi*x/Lx) + Csat - C0)/K0
    exact_expr = 2*Ksat*sin(2*pi*x/Lx)*acos(coord)
    r_expr = conditional(x < lim2, conditional(x > lim1, exact_expr, 0), 0)
    r_exact.interpolate(r_expr)

    # build time stepper
    stepper = PrescribedTransport(meqn, RK4(domain), io,
                                  physics_schemes=physics_schemes)

    stepper.run(0, tmax=tmax)

    error = errornorm(r_exact, meqn.fields("rain"))

    return error


def test_forced_advection(tmpdir):

    tol = 0.1
    error = run_forced_advection(tmpdir)
    assert error < tol, 'The error in the forced advection equation is greater than the permitted tolerance'
