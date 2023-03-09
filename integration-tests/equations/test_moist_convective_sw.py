"""
This tests the moist convective framework of Bouchut et al (2009). The test is
a version of Williamson 5 (flow over orography) where a parcel of water vapour
is advected over topography. Cloud water is produced according to a defined
saturation curve, which is height-dependent. The test checks that cloud is only
produced when the topography is introduced, and checks the amount of cloud
produced is consistent with previous runs.
"""

from os import path
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, exp, conditional, cos,
                       acos)

day = 24*60*60
dt = 300
tmax = 50*day
R = 6371220.
H = 5960.
u_max = 20.
ndumps = 50
dumpfreq = int(tmax / (ndumps*dt))
# moist convective shallow water parameters
q_0 = 3.
alpha = -0.6
tau = 200.
gamma = 5.
q_g = 3
# parameters for set-up
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
theta_c = pi/6.
# initial moisture is cosine blob from deformational test
q_max = 1
b = 0.1
c = 0.9
lamda_1 = 7*pi/6
theta_c = pi/6

def run_moist_convective_sw(tempdir):

    # ----------------------------------------------------------------- #
    # Test case parameters
    # ----------------------------------------------------------------- #

    day = 24*60*60
    dt = 300
    tmax = 50*day
    R = 6371220.
    H = 5960.
    u_max = 20.
    ndumps = 50
    dumpfreq = int(tmax / (ndumps*dt))
    # moist convective shallow water parameters
    q_0 = 3.
    alpha = -0.6
    tau = 200.
    gamma = 5.
    q_g = 3
    # parameters for set-up
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    theta_c = pi/6.
    # initial moisture is cosine blob from deformational test
    q_max = 1
    b = 0.1
    c = 0.9
    lamda_1 = 7*pi/6
    theta_c = pi/6

    # ----------------------------------------------------------------- #
    # Set up model objects
    # ----------------------------------------------------------------- #

    # Domain
    refinements = 3
    mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=1)
    degree = 1
    domain = Domain(mesh, dt, "BDM", degree)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    # Equation
    parameters = ConvectiveMoistShallowWaterParameters(H=H, gamma=gamma,
                                                       tau=tau,q_0=q_0,
                                                       alpha=alpha)
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R

    tracers = [WaterVapour(name='Q', space='DG')]

    # topography
    theta, lamda = latlon_coords(mesh)
    lsq = (lamda - lamda_c)**2
    thsq = (theta - theta_c)**2
    rsq = min_value(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)
    
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr,
                             active_tracers=tracers)

    # Physics schemes
    def sat_func(h):
        return q0 * exp(-alpha*(h-H)/H)

    physics_schemes = [(InstantRain(eqns, sat_func, saturation_variable="D",
                                    vapour_name="Q",
                                    parameters=parameters,
                                    convective_feedback=True),
                        ForwardEuler(domain))]

    # I/O
    diagnostic_fields = [Sum('D', 'topography')]
    output = OutputParameters(dirname=dirname+"/sw")
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')
    Q0 = stepper.fields('Q')

    # interpolate initial conditions; velocity and height are the same as in
    # the dry case
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = equation.parameters.g
    Rsq = R**2
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

    # initial moisture is cosine blob from deformational test
    br = R/4
    r1 = R * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_1))
    q1expr = b + c * (q_max/2)*(1 + cos(pi*r1/br))

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    Q0.interpolate(conditional(r1 < br, 3*q1expr, b))

    # ----------------------------------------------------------------- #
    # Run
    # ----------------------------------------------------------------- #

    stepper.run(t=0, tmax=tmax)

    cloud = stepper.fields("cloud_water")
    vapour = stepper.fields("water_v")

    return vapour, cloud


def test_convective_moist_sw(tmpdir):

    vapour, cloud = run_forced_advection(tmpdir)
    tol = 1
    assert cloud > 0, 'Cloud has not be produced'
    assert 
