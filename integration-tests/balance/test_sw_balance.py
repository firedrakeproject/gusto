from gusto import *
from firedrake import SpatialCoordinate, conditional, Function


def setup_balance(dirname):
    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.                 # planetary radius (m)
    mean_depth = 222.                 # reference depth (m)
    dt = 3600.                        # timestep (s)
    tmax = 10 * dt                    # final time (s)

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, 12, degree=2)

    # Equation
    parameters = ShallowWaterParameters(mesh, H=mean_depth)
    eqns = ShallowWaterEquations

    # I/O
    output = OutputParameters(dirname=dirname, dumpfreq=10)

    # model
    model = SIQNModel(mesh, dt, parameters, eqns, family='BDM')
    model.setup(output)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #
    stepper = model.stepper
    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    # set initial vorticity to be nonzero in a latitude band
    Vcg = model.domain.spaces("H1")
    phi_c = pi/18
    phi_w = 4.5*pi/180
    zeta_s = 3e-5
    x, y, z = SpatialCoordinate(mesh)
    _, phi, _ = lonlatr_from_xyz(x, y, z)
    zeta_expr = conditional(abs(phi-phi_c) > phi_w/2, 0, zeta_s)
    zeta0 = Function(Vcg).interpolate(zeta_expr)

    # calculate corresponding velocity and depth such that initial
    # conditions are nondivergent and div(u_t)=0
    nondivergent_flow(model.equation, zeta0, u0, D0)

    Dbar = Function(D0.function_space()).assign(mean_depth)
    stepper.set_reference_profiles([('D', Dbar)])

    return stepper, tmax, model.domain.spaces("L2")


def run_balance(dirname):

    stepper, tmax, hdiv_space = setup_balance(dirname)
    stepper.run(t=0, tmax=tmax)
    return hdiv_space, stepper.fields("u")


def test_nondivergent_sw(tmpdir):

    dirname = str(tmpdir)
    hdiv_space, u = run_balance(dirname)
    divu = Function(hdiv_space).project(div(u))
    tol = 1e-6
    assert divu.dat.data.max() < tol and abs(divu.dat.data.min()) < tol
