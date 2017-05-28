from gusto import *
from firedrake import as_vector, Constant, sin, PeriodicIntervalMesh, \
    SpatialCoordinate, ExtrudedMesh, Expression
import json
from math import pi

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then advected by a prescribed advection scheme


def setup_theta_limiter(dirname):

    # declare grid shape, with length L and height H
    L = 1000.
    H = 400.
    nlayers = int(H / 20.)
    ncolumns = int(L / 20.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x,z = SpatialCoordinate(mesh)

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=1.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/theta_limiter",
                              dumpfreq=1,
                              dumplist=['u'],
                              perturbation_fields=['theta', 'rho'])
    parameters = CompressibleParameters()

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist)

    # declare initial fields
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vpsi = FunctionSpace(mesh, "CG", 2)
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # make a gradperp
    gradperp = lambda u: as_vector([-u.dx(1), u.dx(0)])

    # Isentropic background state
    Tsurf = 300.
    thetab = Constant(Tsurf)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate initial rho
    compressible_hydrostatic_balance(state, theta_b, rho_b,
                                     solve_for_rho=True)

    # set up bubble
    xc = 200.
    zc = 200.
    rc = 100.
    theta_pert = Function(Vt).interpolate(conditional(sqrt((x - xc) ** 2.0) < rc,
                                                      conditional(sqrt((z - zc) ** 2.0) < rc,
                                                                  Constant(2.0),
                                                                  Constant(0.0)), Constant(0.0)))

    # set up velocity field
    u_max = Constant(1.0)

    psi_expr = - u_max * z

    psi0 = Function(Vpsi).interpolate(psi_expr)
    u0.project(gradperp(psi0))
    theta0.interpolate(theta_b + theta_pert)
    rho0.interpolate(rho_b)

    state.initialise({'u': u0, 'rho': rho0, 'theta': theta0})
    state.set_reference_profiles({'rho': rho_b, 'theta': theta_b})

    # set up advection schemes
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = SUPGAdvection(state, Vt,
                             supg_params={"dg_direction":"horizontal"},
                             equation_form="advective")

    # build advection dictionary
    advection_dict = {}
    advection_dict["u"] = NoAdvection(state, u0, None)
    advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn)
    advection_dict["theta"] = SSPRK3(state, theta0, thetaeqn)

    # build time stepper
    stepper = AdvectionTimestepper(state, advection_dict)

    return stepper, 5.0


def run_theta_limiter(dirname):

    stepper, tmax = setup_theta_limiter(dirname)
    stepper.run(t=0, tmax=tmax)


def test_theta_limiter_setup(tmpdir):

    dirname = str(tmpdir)
    run_theta_limiter(dirname)
