"""
Tests the mean mixing ratio augmentation, which is used for
non-negativity limiting in a conservative transport scheme.
This uses a transport test on the sphere, with non-negativity
and mass conservation checked after a couple of timesteps.

"""

from gusto import *
from firedrake import (
    exp, cos, sin, SpatialCoordinate,
    pi, max_value, assemble, dx
)


def setup_mean_mixing_ratio(dirname):
    # Parameters
    radius = 6371220.      # radius of the sphere, in m
    u_max = 10.            # Transporting velocity
    dt = 1800.             # Timestep size
    tmax = 12*24*60*60.    # Twelve days
    theta_c1 = 0.0         # latitude of first cylinder, in rad
    theta_c2 = 0.0         # latitude of second cylinder, in rad
    lamda_c1 = -pi/4       # longitude of first cylinder, in rad
    lamda_c2 = pi/4        # longitude of second cylinder, in rad
    rho_b = 1.

    mesh = GeneralCubedSphereMesh(radius, 12, degree=2)
    xyz = SpatialCoordinate(mesh)

    # Only use order 1 elements
    domain = Domain(mesh, dt, 'RTCF', 1)

    # get lat lon coordinates
    lamda, theta, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

    tracer_space = 'DG'
    V_tracer = domain.spaces(tracer_space)

    rho_d = ActiveTracer(name='rho_d', space=tracer_space,
                    variable_type=TracerVariableType.density,
                    transport_eqn=TransportEquationType.conservative)

    m_X = ActiveTracer(
                name='m_X', space=tracer_space,
                variable_type=TracerVariableType.mixing_ratio,
                transport_eqn=TransportEquationType.tracer_conservative,
                density_name='rho_d'
            )

    tracers = [rho_d, m_X]

    # Equation
    V = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)

    output = OutputParameters(dirname=dirname)
    io = IO(domain, output)

    augmentation = MeanMixingRatio(domain, eqn, ['m_X'])
    transport_scheme = SSPRK3(domain, augmentation=augmentation, rk_formulation=RungeKuttaFormulation.predictor)

    # Details of transport
    transport_methods = [DGUpwind(eqn, 'rho_d'), DGUpwind(eqn, 'm_X')]

    time_varying_velocity=True
    tau = tmax 

    def u_t(t):
        k = 5.*radius/tau
        u_background = 2*pi*radius/tau
        lamda_prime = lamda - 2*pi*t/tau

        u_zonal = (
            u_background*cos(theta)
            - k*(sin(lamda_prime/2)**2)*sin(2*theta)*(cos(theta)**2)*cos(pi*t/tau)
        )
        u_merid = 0.5*k*sin(lamda_prime)*(cos(theta)**3)*cos(pi*t/tau)

        return xyz_vector_from_lonlatr(u_zonal, u_merid, Constant(0.0), xyz)

    stepper = PrescribedTransport(
    eqn, transport_scheme, io, time_varying_velocity, transport_methods
)

    stepper.setup_prescribed_expr(u_t)

    rho_d_0 = rho_b + 0.5*cos(theta)

    # Slotted cylinders
    m_X_0 = conditional(
                great_arc_angle(lamda, theta, lamda_c1, theta_c1) < 0.5,
                conditional(
                    abs(lamda - lamda_c1) < 1./12.,
                    conditional(theta - theta_c1 < -5./24., 1.0, 0.0),
                    1.0
                ),
                conditional(
                    great_arc_angle(lamda, theta, lamda_c2, theta_c2) < 0.5,
                    conditional(
                        abs(lamda - lamda_c2) < 1./12.,
                        conditional(theta - theta_c2 > 5./24., 1.0, 0.0),
                        1.0
                    ),
                    0.0
                )
            )

    # Initial conditions
    stepper.fields("m_X").interpolate(m_X_0)
    stepper.fields("rho_d").interpolate(rho_d_0)

    rho_X_0 = assemble(stepper.fields("rho_d")*stepper.fields("m_X")*dx)

    return stepper, rho_X_0


def test_mean_mixing_ratio(tmpdir):

    # Setup and run
    dirname = str(tmpdir)

    stepper, rho_X_0 = setup_mean_mixing_ratio(dirname)

    # Run for four timesteps
    dt = 1800.
    stepper.run(t=0, tmax=5*dt)
    rho_d = stepper.fields("rho_d")
    m_X = stepper.fields("m_X")

    rho_X = assemble(rho_d*m_X*dx)
    rho_X_err = np.abs(rho_X - rho_X_0)/rho_X

    # Check that the mixing ratio is non-negative throughout the domain
    #assert all(m_X.dat.data >= 0.0), \
    #    "mean mixing ratio field has not ensured non-negativity"
    assert assemble((abs(m_X) - m_X) * dx) < 1e-14, \
        "mean mixing ratio field has not ensured non-negativity"

    # Confirm mass conservation to a certain tolerance
    assert rho_X_err < 1e-14, "mean mixing ratio field has not ensured mass conservation"
