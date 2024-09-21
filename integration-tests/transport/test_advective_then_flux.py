"""
Tests transport using a Runge-Kutta scheme with an advective-then-flux approach.
This should yield increments that are linear in the divergence (and thus
preserve a constant in divergence-free flow).
"""

from gusto import *
from firedrake import (
    PeriodicRectangleMesh, cos, sin, SpatialCoordinate,
    assemble, dx, pi, as_vector, errornorm, Function
)
import pytest


def setup_advective_then_flux(dirname, desirable_property):

    # ------------------------------------------------------------------------ #
    # Model set up
    # ------------------------------------------------------------------------ #

    # Time parameters
    dt = 2.

    # Domain
    domain_width = 2000.
    ncells_1d = 10.
    mesh = PeriodicRectangleMesh(
        ncells_1d, ncells_1d, domain_width, domain_width, quadrilateral=True
    )
    domain = Domain(mesh, dt, "RTCF", 1)

    # Equation
    V_DG = domain.spaces('DG')
    V_HDiv = domain.spaces("HDiv")
    eqn = ContinuityEquation(domain, V_DG, "rho", Vu=V_HDiv)

    # IO
    output = OutputParameters(dirname=dirname)
    io = IO(domain, output)

    # Transport method
    transport_scheme = SSPRK3(
        domain, rk_formulation=RungeKuttaFormulation.linear, fixed_subcycles=3
    )
    transport_method = DGUpwind(eqn, "rho", advective_then_flux=True)

    # Timestepper
    time_varying = False
    stepper = PrescribedTransport(
        eqn, transport_scheme, io, time_varying, transport_method
    )

    # ------------------------------------------------------------------------ #
    # Initial Conditions
    # ------------------------------------------------------------------------ #

    x, y = SpatialCoordinate(mesh)

    # Density is initially constant for both tests
    rho_0 = 10.0

    # Set the initial state from the configuration choice
    if desirable_property == 'constancy':
        # Divergence free velocity
        num_steps = 5
        psi = Function(domain.spaces('H1'))
        psi_expr = cos(2*pi*x/domain_width)*sin(2*pi*y/domain_width)
        psi.interpolate(psi_expr)
        u_expr = domain.perp(grad(psi_expr))

    elif desirable_property == 'divergence_linearity':
        # Divergent velocity
        num_steps = 1
        u_expr = as_vector([
            cos(2*pi*x/domain_width)*sin(4*pi*y/domain_width),
            -pi*sin(2*pi*x*y/(domain_width)**2)
        ])

    stepper.fields("rho").assign(Constant(rho_0))
    stepper.fields("u").project(u_expr)

    rho_true = Function(V_DG)
    rho_true.assign(stepper.fields("rho"))

    return stepper, rho_true, dt, num_steps


@pytest.mark.parametrize("desirable_property", ["constancy", "divergence_linearity"])
def test_advective_then_flux(tmpdir, desirable_property):

    # Setup and run
    dirname = str(tmpdir)

    stepper, rho_true, dt, num_steps = \
        setup_advective_then_flux(dirname, desirable_property)

    # Run for five timesteps
    stepper.run(t=0, tmax=dt*num_steps)
    rho = stepper.fields("rho")

    # Check for divergence-linearity/constancy
    assert errornorm(rho, rho_true) < 2e-13, \
        "advective-then-flux form is not yielding the correct answer"

    # Check for conservation
    mass_initial = assemble(rho_true*dx)
    mass_final = assemble(rho*dx)
    assert abs(mass_final - mass_initial) < 1e-14, \
        "advective-then-flux form is not conservative"
