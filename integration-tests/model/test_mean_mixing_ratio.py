"""
Tests the mean mixing ratio augmentation. Here a mixing ratio is transported
conservatively along with the dry density. The mean mixing ratio augmentation
should ensure that the mixing ratio remains non-negative. To test this,
we start with a mixing ratio field that has some negative values, then
see if the limiter ensures that these values become non-negative.
"""

from gusto import *
from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, exp, cos, sin, SpatialCoordinate,
    pi, min_value, as_vector
)


def setup_mean_mixing_ratio(dirname):

    # Domain
    Lx = 2000.
    Hz = 2000.

    # Time parameters
    dt = 2.
    tmax = 2000.

    nlayers = 10.  # horizontal layers
    columns = 10.  # number of columns

    # Define the spaces for the tracers
    rho_d_space = 'DG'
    m_X_space = 'DG'

    period_mesh = PeriodicIntervalMesh(columns, Lx)
    mesh = ExtrudedMesh(period_mesh, layers=nlayers, layer_height=Hz/nlayers)
    domain = Domain(mesh, dt, "CG", 1)
    x, z = SpatialCoordinate(mesh)

    V_rho = domain.spaces(rho_d_space)
    V_m_X = domain.spaces(m_X_space)

    m_X = ActiveTracer(name='m_X', space=m_X_space,
                       variable_type=TracerVariableType.mixing_ratio,
                       transport_eqn=TransportEquationType.tracer_conservative,
                       density_name='rho_d')

    rho_d = ActiveTracer(name='rho_d', space=rho_d_space,
                         variable_type=TracerVariableType.density,
                         transport_eqn=TransportEquationType.conservative)

    # Define m_X first to test that the tracers will be
    # automatically re-ordered such that the density field
    # is indexed before the mixing ratio.
    tracers = [m_X, rho_d]

    # Equation
    V = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)

    # IO
    output = OutputParameters(dirname=dirname)
    io = IO(domain, output)

    augmentation = MeanMixingRatio(domain, eqn, ['m_X'])
    transport_scheme = SSPRK3(
        domain, augmentation=augmentation, rk_formulation=RungeKuttaFormulation.predictor
    )
    transport_methods = [DGUpwind(eqn, "m_X"), DGUpwind(eqn, "rho_d")]

    # Timestepper
    time_varying = True
    stepper = PrescribedTransport(
        eqn, transport_scheme, io, time_varying, transport_methods
    )

    # Initial Conditions
    xc1 = 5.*Lx/8.
    zc1 = Hz/2.
    xc2 = 3.*Lx/8.
    zc2 = Hz/2.

    def l2_dist(xc, zc):
        return min_value(abs(x-xc), Lx-abs(x-xc))**2 + (z-zc)**2

    lc = 2.*Lx/25.
    f0 = 0.5
    rho_b = 0.5

    g1 = f0*exp(-l2_dist(xc1, zc1)/(lc**2))
    g2 = f0*exp(-l2_dist(xc2, zc2)/(lc**2))

    rho_d_0 = rho_b + g1 + g2

    # Make the mixing ratio contain negative values to
    # test that the limiter is working.
    m0 = -0.05
    m_X_0 = m0 + 0.5*z/Hz

    # Set up the divergent, time-varying, velocity field
    U = Lx/tmax
    W = U/10.

    def u_t(t):
        xd = x - U*t
        u = U - (W*pi*Lx/Hz)*cos(pi*t/tmax)*cos(2*pi*xd/Lx)*cos(pi*z/Hz)
        w = 2*pi*W*cos(pi*t/tmax)*sin(2*pi*xd/Lx)*sin(pi*z/Hz)

        u_expr = as_vector((u, w))

        return u_expr

    stepper.setup_prescribed_expr(u_t)

    stepper.fields("m_X").interpolate(m_X_0)
    stepper.fields("rho_d").interpolate(rho_d_0)
    stepper.fields("u").project(u_t(0))

    m_X_init = Function(V_m_X)
    rho_d_init = Function(V_rho)

    m_X_init.assign(stepper.fields("m_X"))
    rho_d_init.assign(stepper.fields("rho_d"))

    return stepper, m_X_init, rho_d_init


def test_mean_mixing_ratio(tmpdir):

    # Setup and run
    dirname = str(tmpdir)

    stepper, m_X_0, rho_d_0 = \
        setup_mean_mixing_ratio(dirname)

    # Run for two timesteps
    stepper.run(t=0, tmax=2)
    m_X = stepper.fields("m_X")

    # Check that the mixing ratio has no negative values
    assert all(m_X.dat.data >= 0.0), "mean mixing ratio field has not ensured non-negativity"
