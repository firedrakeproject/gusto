"""
This tests the linear shallow water equations.
The test uses the initial conditions of the Williamson 2 test and checks the
errors in the fields.
"""

from os import path
from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi
from netCDF4 import Dataset


def setup_sw(dirname):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    dt = 3600.
    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 2000.
    day = 24.*60.*60.

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements, degree=3)
    x = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, "BDM", degree=1)

    # Equation
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D')]
    output = OutputParameters(dirname=dirname+"/sw_linear_w2", dumpfreq=12)
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transport_schemes = [ForwardEuler(domain, "D")]
    transport_methods = []

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transport_schemes, transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = parameters.g
    Dexpr = H-((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    return stepper, 2*day


def run_sw(dirname):

    stepper, tmax = setup_sw(dirname)
    stepper.run(t=0, tmax=tmax)


def test_sw_linear(tmpdir):
    dirname = str(tmpdir)
    run_sw(dirname)
    filename = path.join(dirname, "sw_linear_w2/diagnostics.nc")
    data = Dataset(filename, "r")

    Derr = data.groups["D_error"]
    D = data.groups["D"]
    Dl2 = Derr["l2"][-1]/D["l2"][0]
    uerr = data.groups["u_error"]
    u = data.groups["u"]
    ul2 = uerr["l2"][-1]/u["l2"][0]

    assert Dl2 < 3.e-3
    assert ul2 < 6.e-2
