from os import path
from gusto import *
from firedrake import as_vector, Constant, PeriodicIntervalMesh, \
    SpatialCoordinate, ExtrudedMesh, FunctionSpace, Function, \
    conditional, sqrt
from netCDF4 import Dataset

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then advected by a prescribed advection scheme


def setup_theta_limiter(dirname):

    # declare grid shape, with length L and height H
    L = 400.
    H = 400.
    nlayers = int(H / 10.)
    ncolumns = int(L / 10.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x, z = SpatialCoordinate(mesh)

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=1.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/theta_limiter",
                              dumpfreq=5,
                              dumplist=['u'],
                              perturbation_fields=['theta'])
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

    # set up bubble
    xc = 200.
    zc = 200.
    rc = 100.
    theta_pert = Function(Vt).interpolate(conditional(sqrt((x - xc) ** 2.0) < rc,
                                                      conditional(sqrt((z - zc) ** 2.0) < rc,
                                                                  Constant(2.0),
                                                                  Constant(0.0)), Constant(0.0)))

    # set up velocity field
    u_max = Constant(10.0)

    psi_expr = - u_max * z

    psi0 = Function(Vpsi).interpolate(psi_expr)
    u0.project(gradperp(psi0))
    theta0.interpolate(theta_b + theta_pert)

    state.initialise([('u', u0), ('rho', rho0), ('theta', theta0)])
    state.set_reference_profiles([('theta', theta_b)])

    # set up advection schemes
    rhoeqn = EmbeddedDGAdvection(state, Vr, equation_form="continuity")
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")

    # build advection dictionary
    advected_fields = []
    advected_fields.append(('u', NoAdvection(state, u0, None)))
    advected_fields.append(('rho', SSPRK3(state, rho0, rhoeqn)))
    advected_fields.append(('theta', SSPRK3(state, theta0, thetaeqn, limiter=ThetaLimiter(thetaeqn.space))))

    # build time stepper
    stepper = AdvectionTimestepper(state, advected_fields)

    return stepper, 40.0


def run_theta_limiter(dirname):

    stepper, tmax = setup_theta_limiter(dirname)
    stepper.run(t=0, tmax=tmax)


def test_theta_limiter_setup(tmpdir):

    dirname = str(tmpdir)
    run_theta_limiter(dirname)
    filename = path.join(dirname, "theta_limiter/diagnostics.nc")
    data = Dataset(filename, "r")

    theta = data.groups["theta"]
    max_theta = theta.variables["max"]
    min_theta = theta.variables["min"]

    assert max_theta[-1] <= max_theta[0]
    assert min_theta[-1] >= min_theta[0]
