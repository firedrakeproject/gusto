"""
Tests the conservative transport of a mixing ratio and dry density, both when
they are defined on the same and different function spaces. This checks
that there is conservation of the total species mass (dry density times the
mixing ratio) and that there is consistency (a constant field will remain
constant).
"""

from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, exp, cos, sin, SpatialCoordinate, \
    assemble, dx, FunctionSpace, pi, min_value, as_vector, BrokenElement, errornorm
import pytest


def setup_conservative_transport(dirname, pair_of_spaces, property):

    # Domain
    Lx = 2000.
    Hz = 2000.

    # Time parameters
    dt = 2.
    tmax = 2000.

    nlayers = 10.  # horizontal layers
    columns = 10.  # number of columns

    # Define the spaces for the tracers
    if pair_of_spaces == 'same_order_1':
        rho_d_space = 'DG'
        m_X_space = 'DG'
        space_order = 1
    elif pair_of_spaces == 'diff_order_0':
        rho_d_space = 'DG'
        m_X_space = 'theta'
        space_order = 0
    elif pair_of_spaces == 'diff_order_1':
        rho_d_space = 'DG'
        m_X_space = 'theta'
        space_order = 1

    period_mesh = PeriodicIntervalMesh(columns, Lx)
    mesh = ExtrudedMesh(period_mesh, layers=nlayers, layer_height=Hz/nlayers)
    domain = Domain(mesh, dt, "CG", space_order)
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
    output = OutputParameters(dirname=dirname)

    io = IO(domain, output)

    # Set up the divergent, time-varying, velocity field
    U = Lx/tmax
    W = U/10.

    def u_t(t):
        xd = x - U*t
        u = U - (W*pi*Lx/Hz)*cos(pi*t/tmax)*cos(2*pi*xd/Lx)*cos(pi*z/Hz)
        w = 2*pi*W*cos(pi*t/tmax)*sin(2*pi*xd/Lx)*sin(pi*z/Hz)

        u_expr = as_vector((u, w))

        return u_expr

    # Specify locations of the two Gaussians
    xc1 = 5.*Lx/8.
    zc1 = Hz/2.

    xc2 = 3.*Lx/8.
    zc2 = Hz/2.

    def l2_dist(xc, zc):
        return min_value(abs(x-xc), Lx-abs(x-xc))**2 + (z-zc)**2

    lc = 2.*Lx/25.
    m0 = 0.02

    # Set the initial state from the configuration choice
    if property == 'conservation':
        f0 = 0.05

        rho_t = 0.5
        rho_b = 1.

        rho_d_0 = rho_b + z*(rho_t-rho_b)/Hz

        g1 = f0*exp(-l2_dist(xc1, zc1)/(lc**2))
        g2 = f0*exp(-l2_dist(xc2, zc2)/(lc**2))

        m_X_0 = m0 + g1 + g2
    else:
        f0 = 0.5
        rho_b = 0.5

        g1 = f0*exp(-l2_dist(xc1, zc1)/(lc**2))
        g2 = f0*exp(-l2_dist(xc2, zc2)/(lc**2))

        rho_d_0 = rho_b + g1 + g2

        # Constant mass field
        m_X_0 = m0 + 0*x

    if pair_of_spaces == 'diff_order_0':
        VCG1 = FunctionSpace(mesh, 'CG', 1)
        VDG1 = domain.spaces('DG1_equispaced')

        suboptions = {'rho_d': RecoveryOptions(embedding_space=VDG1,
                                               recovered_space=VCG1,
                                               project_low_method='recover',
                                               boundary_method=BoundaryMethod.taylor),
                      'm_X': ConservativeRecoveryOptions(embedding_space=VDG1,
                                                         recovered_space=VCG1,
                                                         boundary_method=BoundaryMethod.taylor,
                                                         rho_name='rho_d',
                                                         orig_rho_space=V_rho)
                      }
    elif pair_of_spaces == 'diff_order_1':
        Vt_brok = FunctionSpace(mesh, BrokenElement(V_m_X.ufl_element()))
        suboptions = {'rho_d': EmbeddedDGOptions(embedding_space=Vt_brok),
                      'm_X': ConservativeEmbeddedDGOptions(rho_name='rho_d',
                                                           orig_rho_space=V_rho)}
    else:
        suboptions = {}

    opts = MixedFSOptions(suboptions=suboptions)

    transport_scheme = SSPRK3(domain, options=opts, increment_form=False)
    transport_methods = [DGUpwind(eqn, "m_X"), DGUpwind(eqn, "rho_d")]

    # Timestepper
    stepper = PrescribedTransport(eqn, transport_scheme, io, transport_methods, prescribed_transporting_velocity=u_t)

    # Initial Conditions
    stepper.fields("m_X").interpolate(m_X_0)
    stepper.fields("rho_d").interpolate(rho_d_0)
    u0 = stepper.fields("u")
    u0.project(u_t(0))

    m_X_init = Function(V_m_X)
    rho_d_init = Function(V_rho)

    m_X_init.assign(stepper.fields("m_X"))
    rho_d_init.assign(stepper.fields("rho_d"))

    return stepper, m_X_init, rho_d_init


@pytest.mark.parametrize("pair_of_spaces", ["same_order_1", "diff_order_0", "diff_order_1"])
@pytest.mark.parametrize("property", ["consistency", "conservation"])
def test_conservative_transport(tmpdir, pair_of_spaces, property):

    # Setup and run
    dirname = str(tmpdir)

    stepper, m_X_0, rho_d_0 = setup_conservative_transport(dirname, pair_of_spaces, property)

    # Run for five timesteps
    stepper.run(t=0, tmax=10)
    m_X = stepper.fields("m_X")
    rho_d = stepper.fields("rho_d")

    # Perform the check
    if property == 'consistency':
        assert errornorm(m_X_0, m_X) < 1e-13, "conservative transport is not consistent"
    else:
        rho_X_init = assemble(m_X_0*rho_d_0*dx)
        rho_X_final = assemble(m_X*rho_d*dx)
        assert abs((rho_X_init - rho_X_final)/rho_X_init) < 1e-14, "conservative transport is not conservative"
