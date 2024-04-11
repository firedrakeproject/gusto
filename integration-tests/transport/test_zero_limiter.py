"""
Testing the zero limiter. Without the limiter small amounts of negative cloud
are produced after 1 timestep, so the first test will fail. The second test
applies the limiter to the cloud field only, and should pass. In the third test
the limiter is applied to both cloud and rain and to see if the application of
the limiter to rain stops the limiter working on cloud.
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, sin, cos, exp


def setup_zero_limiter(dirname, limiter=False, rain=False):

    # ----------------------------------------------------------------- #
    # Test case parameters
    # ----------------------------------------------------------------- #

    dt = 3000
    tmax = 1*dt
    ref = 3
    R = 6371220.
    u_max = 20
    phi_0 = 3e4
    epsilon = 1/300
    theta_0 = epsilon*phi_0**2
    g = 9.80616
    H = phi_0/g
    xi = 0
    q0 = 200
    beta2 = 10

    # ----------------------------------------------------------------- #
    # Set up model objects
    # ----------------------------------------------------------------- #

    # Domain
    mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref, degree=2)
    degree = 1

    domain = Domain(mesh, dt, 'BDM', degree)

    x = SpatialCoordinate(mesh)

    # Equations
    parameters = ShallowWaterParameters(H=H, g=g)
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R

    if rain:
        tracers = [WaterVapour(space='DG'), CloudWater(space='DG'), Rain(space='DG')]
    else:
        tracers = [WaterVapour(space='DG'), CloudWater(space='DG')]

    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                 u_transport_option='vector_advection_form',
                                 thermal=True, active_tracers=tracers)

    output = OutputParameters(dirname=dirname, dumpfreq=1)

    io = IO(domain, output)

    # ------------------------------------------------------------------------ #
    # Set up physics and transport schemes
    # ------------------------------------------------------------------------ #

    # Saturation function
    def sat_func(x_in):
        D = x_in.split()[1]
        b = x_in.split()[2]
        return q0/(g*D) * exp(20*(1 - b/g))

    # Feedback proportionality is dependent on h and b
    def gamma_v(x_in):
        D = x_in.split()[1]
        b = x_in.split()[2]
        return (1 + 10*(20*q0/g*D * exp(20*(1 - b/g))))**(-1)

    transport_methods = [DGUpwind(eqns, field_name) for field_name in eqns.field_names]

    linear_solver = ThermalSWSolver(eqns)

    zerolimiter = ZeroLimiter(domain.spaces('DG'))
    DG1limiter = DG1Limiter(domain.spaces('DG'))

    if rain:
        physics_sublimiters = {'cloud_water': zerolimiter,
                               'rain': zerolimiter}
    else:
        physics_sublimiters = {'cloud_water': zerolimiter}

    physics_limiter = MixedFSLimiter(eqns, physics_sublimiters)

    sat_adj = SWSaturationAdjustment(eqns, sat_func,
                                     time_varying_saturation=True,
                                     parameters=parameters,
                                     thermal_feedback=True,
                                     beta2=beta2, gamma_v=gamma_v,
                                     time_varying_gamma_v=True)

    if limiter:
        physics_schemes = [(sat_adj, ForwardEuler(domain, limiter=physics_limiter))]
    else:
        physics_schemes = [(sat_adj, ForwardEuler(domain))]

    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "D"),
                          SSPRK3(domain, "b", limiter=DG1limiter),
                          SSPRK3(domain, "water_vapour", limiter=DG1limiter),
                          SSPRK3(domain, "cloud_water", limiter=DG1limiter),
                          ]
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                      transport_methods,
                                      linear_solver=linear_solver,
                                      physics_schemes=physics_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")
    v0 = stepper.fields("water_vapour")

    _, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])

    uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, x)
    g = parameters.g
    w = Omega*R*u_max + (u_max**2)/2
    sigma = w/10

    Dexpr = H - (1/g)*(w + sigma)*((sin(phi))**2)

    numerator = theta_0 + sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))

    denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2

    theta = numerator/denominator

    bexpr = parameters.g * (1 - theta)

    initial_msat = q0/(g*Dexpr) * exp(20*theta)
    vexpr = (1 - xi) * initial_msat

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)
    v0.interpolate(vexpr)

    # Set reference profiles
    Dbar = Function(D0.function_space()).assign(H)
    bbar = Function(b0.function_space()).interpolate(bexpr)
    stepper.set_reference_profiles([('D', Dbar), ('b', bbar)])

    return stepper, tmax


def test_without_limiter(tmpdir):

    # Setup and run verison without limiter
    dirname = str(tmpdir)

    stepper_without_limiter, tmax = setup_zero_limiter(dirname)

    stepper_without_limiter.run(t=0, tmax=tmax)

    cloud_without_limiter = stepper_without_limiter.fields('cloud_water')

    assert cloud_without_limiter.dat.data.min() < 0, "The minimum of cloud is negative"


def test_with_limiter(tmpdir):

    # Setup and run verison with limiter
    dirname = str(tmpdir)

    stepper_with_limiter, tmax = setup_zero_limiter(dirname, limiter=True)

    stepper_with_limiter.run(t=0, tmax=tmax)

    cloud_with_limiter = stepper_with_limiter.fields('cloud_water')

    assert cloud_with_limiter.dat.data.min() >= 0, "Application of the limiter has not stopped negative values in cloud"


def test_limiter_with_rain(tmpdir):

    # Setup and run verison with limiter where rain is also limited
    dirname = str(tmpdir)

    stepper_with_rain_limiter, tmax = setup_zero_limiter(dirname, limiter=True,
                                                         rain=True)

    stepper_with_rain_limiter.run(t=0, tmax=tmax)

    cloud_with_rain_limiter = stepper_with_rain_limiter.fields('cloud_water')

    assert cloud_with_rain_limiter.dat.data.min() >= 0, "Using the limiter on both cloud and rain has not stopped negatives in cloud"
