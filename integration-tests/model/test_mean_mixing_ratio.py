"""
Tests the mean mixing ratio augmentation, which is used for
non-negativity limiting in a conservative transport scheme.
A few timesteps are taken in the terminator toy test, with
non-negativity and mass conservation checked.

"""

from gusto import *
from firedrake import (
    exp, cos, sin, SpatialCoordinate,
    pi, max_value, assemble, dx
)


def setup_mean_mixing_ratio(dirname):
    dt = 450.
    tau = 12.*24.*60.*60.  # time period of reversible wind, in s
    radius = 6371220.      # radius of the sphere, in m
    theta_cr = pi/9.       # central latitude of first reaction rate, in rad
    lamda_cr = -pi/3.      # central longitude of first reaction rate, in rad
    theta_c1 = 0.          # central latitude of first chemical blob, in rad
    theta_c2 = 0.          # central latitude of second chemical blob, in rad
    lamda_c1 = -pi/4.      # central longitude of first chemical blob, in rad
    lamda_c2 = pi/4.       # central longitude of second chemical blob, in rad
    rho_b = 1              # Base dry density
    g_max = 0.5            # Maximum amplitude of Gaussian density perturbations
    b0 = 5                 # Controls the width of the chemical blobs

    mesh = GeneralCubedSphereMesh(radius, 12, degree=2)
    xyz = SpatialCoordinate(mesh)

    # Only use order 1 elements
    domain = Domain(mesh, dt, 'RTCF', 1)

    # get lat lon coordinates
    lamda, theta, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

    # Define co-located tracers of the dry density and the two species
    rho_d = ActiveTracer(name='rho_d', space='DG',
                         variable_type=TracerVariableType.density,
                         transport_eqn=TransportEquationType.conservative)

    X_tracer = ActiveTracer(name='X_tracer', space='DG',
                            variable_type=TracerVariableType.mixing_ratio,
                            transport_eqn=TransportEquationType.tracer_conservative,
                            density_name='rho_d')

    X2_tracer = ActiveTracer(name='X2_tracer', space='DG',
                             variable_type=TracerVariableType.mixing_ratio,
                             transport_eqn=TransportEquationType.tracer_conservative,
                             density_name='rho_d')

    # Define the mixing ratios first to test that the tracers will be
    # automatically re-ordered such that the density field
    # is indexed before the mixing ratio.
    tracers = [X_tracer, X2_tracer, rho_d]

    # Equation
    V = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)

    output = OutputParameters(dirname=dirname)
    io = IO(domain, output)

    k1 = max_value(0, sin(theta)*sin(theta_cr) + cos(theta)*cos(theta_cr)*cos(lamda-lamda_cr))
    k2 = 1

    mixed_phys_limiter = MixedFSLimiter(
        eqn,
        {'rho_d': ZeroLimiter(domain.spaces('DG')),
         'X_tracer': ZeroLimiter(domain.spaces('DG')),
         'X2_tracer': ZeroLimiter(domain.spaces('DG'))}
    )

    # Using the analytical forcing from Appendix D of Lauritzen et. al.
    physics_schemes = [(TerminatorToy(eqn, k1=k1, k2=k2, species1_name='X_tracer',
                        species2_name='X2_tracer', analytical_formulation=True),
                        ForwardEuler(domain, limiter=mixed_phys_limiter))]

    X, Y, Z = xyz
    X1, Y1, Z1 = xyz_from_lonlatr(lamda_c1, theta_c1, radius)
    X2, Y2, Z2 = xyz_from_lonlatr(lamda_c2, theta_c2, radius)

    g1 = g_max*exp(-(b0/(radius**2))*((X-X1)**2 + (Y-Y1)**2 + (Z-Z1)**2))
    g2 = g_max*exp(-(b0/(radius**2))*((X-X2)**2 + (Y-Y2)**2 + (Z-Z2)**2))

    rho_expr = rho_b + g1 + g2

    X_T_0 = 4e-6
    r = k1/(4*k2)
    D_val = sqrt(r**2 + 2*X_T_0*r)

    # Initial condition for each species
    X_0 = D_val - r
    X2_0 = 0.5*(X_T_0 - D_val + r)

    def u_t(t):
        k = 10*radius/tau

        u_zonal = (
            k*(sin(lamda - 2*pi*t/tau)**2)*sin(2*theta)*cos(pi*t/tau)
            + ((2*pi*radius)/tau)*cos(theta)
        )
        u_merid = k*sin(2*(lamda - 2*pi*t/tau))*cos(theta)*cos(pi*t/tau)

        return xyz_vector_from_lonlatr(u_zonal, u_merid, Constant(0.0), xyz)

    augmentation = MeanMixingRatio(domain, eqn, ['X_tracer', 'X2_tracer'])
    transport_scheme = SSPRK3(domain, augmentation=augmentation, rk_formulation=RungeKuttaFormulation.predictor)
    transport_method = [DGUpwind(eqn, 'rho_d'), DGUpwind(eqn, 'X_tracer'), DGUpwind(eqn, 'X2_tracer')]

    time_varying_velocity = True
    stepper = SplitPrescribedTransport(eqn, transport_scheme, io,
                                       time_varying_velocity,
                                       spatial_methods=transport_method,
                                       physics_schemes=physics_schemes)

    stepper.setup_prescribed_expr(u_t)

    # Initial conditions
    stepper.fields("rho_d").interpolate(rho_expr)
    stepper.fields("X_tracer").interpolate(X_0)
    stepper.fields("X2_tracer").interpolate(X2_0)

    X_sum = assemble(stepper.fields("rho_d")*stepper.fields("X_tracer")*dx)
    X2_sum = assemble(stepper.fields("rho_d")*stepper.fields("X2_tracer")*dx)
    XT_init = X_sum + 2*X2_sum

    return stepper, XT_init


def test_mean_mixing_ratio(tmpdir):

    # Setup and run
    dirname = str(tmpdir)

    stepper, XT_init = setup_mean_mixing_ratio(dirname)

    # Run for four timesteps
    stepper.run(t=0, tmax=1800.)
    rho_d = stepper.fields("rho_d")
    X_tracer = stepper.fields("X_tracer")
    X2_tracer = stepper.fields("X2_tracer")

    X_sum = assemble(rho_d*X_tracer*dx)
    X2_sum = assemble(rho_d*X2_tracer*dx)
    XT_sum = X_sum + 2*X2_sum
    td_err = np.abs(XT_init - XT_sum)/XT_init

    # Check that all the fields are non-negative
    assert all(X_tracer.dat.data >= 0.0) and all(X2_tracer.dat.data >= 0.0), \
        "mean mixing ratio field has not ensured non-negativity"

    # Confirm mass conservation to a certain tolerance
    assert td_err < 1e-14, "mean mixing ratio field has not ensured mass conservation"
