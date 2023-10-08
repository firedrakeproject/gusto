"""
This tests the thermal shallow water equations using the initial conditions of
Test 1 from Zerroukat and Allen (2015), which is based on Williamson 2. Initial
conditions for velocity, height and buoyancy are prescribed and the final fields
are checked against these.

"""

from os import path
from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, pi, sin, cos
from netCDF4 import Dataset

R = 6371220.
g = 9.80616
day = 24.*60.*60.
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
phi_0 = 3e4
epsilon = 1/300
theta_0 = epsilon*phi_0**2
H = phi_0/g


def setup_sw(dirname, dt, u_transport_option):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    refinements = 3

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements,
                                 degree=2)
    domain = Domain(mesh, dt, family="BDM", degree=1)
    x = SpatialCoordinate(mesh)

    # Equation
    parameters = ShallowWaterParameters(H=H, g=g)
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                 u_transport_option=u_transport_option,
                                 thermal=True)
    # I/O
    diagnostic_fields = [SteadyStateError('D'),
                         SteadyStateError('u'),
                         SteadyStateError('b')]

    output = OutputParameters(dirname=dirname+"/sw", dumplist_latlon=['D', 'D_error'])
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    return domain, eqns, io


def set_up_initial_conditions(domain, equation, stepper):

    # interpolate initial conditions
    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")

    g = equation.parameters.g
    Omega = equation.parameters.Omega

    x = SpatialCoordinate(domain.mesh)
    _, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])

    uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, x)
    w = Omega*R*u_max + (u_max**2)/2
    sigma = 0
    Dexpr = H - (1/g)*(w + sigma)*((sin(phi))**2)

    numerator = theta_0 - sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))

    denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2

    theta = numerator/denominator

    bexpr = g * (1 - theta)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])


def check_results(dirname):
    filename = path.join(dirname, "sw/diagnostics.nc")
    data = Dataset(filename, "r")

    Derr = data.groups["D_error"]
    D = data.groups["D"]
    Dl2 = Derr["l2"][-1]/D["l2"][0]
    assert Dl2 < 5.e-4

    uerr = data.groups["u_error"]
    u = data.groups["u"]
    ul2 = uerr["l2"][-1]/u["l2"][0]
    assert ul2 < 5.e-3

    berr = data.groups["b_error"]
    b = data.groups["b"]
    bl2 = berr["l2"][-1]/b["l2"][0]
    assert bl2 < 5.e-5


def test_sw_ssprk3(tmpdir):

    u_transport_option = "vector_advection_form"

    dirname = str(tmpdir)
    dt = 100
    domain, eqns, io = setup_sw(dirname, dt, u_transport_option)

    transport_methods = [DGUpwind(eqns, 'u'),
                         DGUpwind(eqns, 'D'),
                         DGUpwind(eqns, 'b')]

    stepper = Timestepper(eqns, SSPRK3(domain), io, transport_methods)

    # Initial conditions
    set_up_initial_conditions(domain, eqns, stepper)

    # Run
    stepper.run(t=0, tmax=0.01*day)

    check_results(dirname)
