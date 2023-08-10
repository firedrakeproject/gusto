"""
This tests the source/sink
"""

from gusto import *
from firedrake import (as_vector, PeriodicSquareMesh, SpatialCoordinate,
                       sqrt, sin, pi, assemble, Constant)
import pytest

def run_source_sink(dirname, process, time_varying):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # set up mesh and domain
    L = 10
    nx = 10
    mesh = PeriodicSquareMesh(nx, nx, L, quadrilateral=True)
    dt = 0.1
    tmax = 5*dt
    domain = Domain(mesh, dt, "RTCF", 1)
    x, y = SpatialCoordinate(mesh)

    # Equation
    V = domain.spaces('DG')
    eqn = AdvectionEquation(domain, V, "ash")

    # I/O
    output = OutputParameters(dirname=dirname+"/source_sink",
                              dumpfreq=1)
    diagnostic_fields = [CourantNumber()]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)
    transport_method = [DGUpwind(eqn, "ash")]

    # Physics scheme --------------------------------------------------------- #
    # Source is a Lorentzian centred on a point
    centre_x = L / 4.0
    centre_y = 3*L / 4.0
    width = L / 8.0
    dist_x = periodic_distance(x, centre_x, L)
    dist_y = periodic_distance(y, centre_y, L)
    dist = sqrt(dist_x**2 + dist_y**2)
    # Lorentzian function
    basic_expression = width / (dist**2 + width**2)

    if process == 'source':
        basic_expression = -basic_expression

    def time_varying_expression(t):
        return 2*basic_expression*sin(pi*t/(2.0*tmax))

    if time_varying:
        expression = time_varying_expression
    else:
        expression = basic_expression

    physics_parametrisations = [SourceSink(eqn, 'ash', expression, time_varying)]

    # Time stepper
    stepper = PrescribedTransport(eqn, SSPRK3(domain), io, transport_method,
                                  physics_parametrisations=physics_parametrisations)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    ash0 = stepper.fields("ash")

    if process == "source":
        # Start with no ash
        background_ash = Constant(0.0)
    elif process == "sink":
        # Take ash away
        background_ash = Constant(100.0)
    ash0.interpolate(background_ash)
    initial_ash = Function(V).assign(ash0)

    # Constant wind
    u = stepper.fields("u")
    u.project(as_vector([1.0, 0.0]))

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)
    return stepper, initial_ash


@pytest.mark.parametrize("process", ["source", "sink"])
@pytest.mark.parametrize("time_varying", [False, True])
def test_source_sink(tmpdir, process, time_varying):
    dirname = str(tmpdir)
    stepper, initial_ash = run_source_sink(dirname, process, time_varying)
    final_ash = stepper.fields("ash")

    initial_total_ash = assemble(initial_ash*dx)
    final_total_ash = assemble(final_ash*dx)

    tol = 1.0
    if process == "source":
        assert final_total_ash > initial_total_ash + tol, \
            "Source process does not appear to have created tracer"
    else:
        assert final_total_ash < initial_total_ash - tol, \
            "Sink process does not appear to have removed tracer"
