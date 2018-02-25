from os import path
from gusto import *
from firedrake import as_vector, Constant, PeriodicIntervalMesh, \
    SpatialCoordinate, ExtrudedMesh, FunctionSpace, Function, \
    conditional, sqrt, FiniteElement, TensorProductElement, BrokenElement
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from netCDF4 import Dataset

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then advected by a prescribed advection scheme


def setup_limiters(dirname):

    # declare grid shape
    L = 400.
    H = L
    ncolumns = int(L / 10.)
    nlayers = ncolumns

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x, z = SpatialCoordinate(mesh)

    fieldlist = ['u']
    timestepping = TimesteppingParameters(dt=1.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/limiting",
                              dumpfreq=5,
                              dumplist=['u'],
                              perturbation_fields=['theta0v', 'theta0h', 'theta1'])
    parameters = CompressibleParameters()

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist)

    # make elements
    # v is continuous in vertical, h is horizontal
    cell = mesh._base_mesh.ufl_cell().cellname()
    DG0_element = FiniteElement("DG", cell, 0)
    CG1_element = FiniteElement("CG", cell, 1)
    DG1_element = FiniteElement("DG", cell, 1)
    v0_element = TensorProductElement(DG0_element, CG1_element)
    h0_element = TensorProductElement(CG1_element, DG0_element)
    v1_element = TensorProductElement(DG1_element, CG2_element)

    # spaces
    Vpsi = FunctionSpace(mesh, "CG", 2)
    VDG1 = FunctionSpace(mesh, "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    V0v = FunctionSpace(mesh, v0_element)
    V0h = FunctionSpace(mesh, h0_element)
    V1v = FunctionSpace(mesh, v1_element)

    V0v_brok = FunctionSpace(mesh, BrokenElement(V0v.ufl_element()))
    V0h_brok = FunctionSpace(mesh, BrokenElement(V0h.ufl_element()))

    v0_spaces = (VDG1, VCG1, V0v_brok)
    h0_spaces = (VDG1, VCG1, V0h_brok)

    # declare initial fields
    u0 = state.fields("u")
    theta0v = state.fields("theta0v", V0v)
    theta0h = state.fields("theta0h", V0h)
    theta1 = state.fields("theta1", V1v)

    # make a gradperp
    gradperp = lambda u: as_vector([-u.dx(1), u.dx(0)])

    # Isentropic background state
    Tsurf = 300.
    thetab = Constant(Tsurf)
    theta_b1 = Function(V1v).interpolate(thetab)
    theta_b0v = Function(V0v).interpolate(thetab)
    theta_b0h = Function(V0h).interpolate(thetab)

    # set up bubble
    xc = 200.
    zc = 200.
    rc = 100.
    theta_expr = conditional(sqrt((x - xc) ** 2.0) < rc,
                             conditional(sqrt((z - zc) ** 2.0) < rc,
                                         Constant(2.0),
                                         Constant(0.0)), Constant(0.0))
    theta_pert1 = Function(V1v).interpolate(theta_expr)
    theta_pert0v = Function(V0v).interpolate(theta_expr)
    theta_pert0h = Function(V0h).interpolate(theta_expr)

    # set up velocity field
    u_max = Constant(10.0)

    psi_expr = - u_max * z

    psi0 = Function(Vpsi).interpolate(psi_expr)
    u0.project(gradperp(psi0))
    theta0v.interpolate(theta_b0v + theta_pert0v)
    theta0h.interpolate(theta_b0h + theta_pert0h)
    theta1.interpolate(theta_b1 + theta_pert1)

    state.initialise([('u', u0), ('theta1', theta1), ('theta0v', theta0v), ('theta0h', theta0h)])
    state.set_reference_profiles([('theta1', theta_b1), ('theta0v', theta_b0v), ('theta0h', theta_b0h)])

    # set up advection schemes
    thetaeqn1 = EmbeddedDGAdvection(state, V1v, equation_form="advective")
    thetaeqn0v = EmbeddedDGAdvection(state, V0v, equation_form="advective", recovered_spaces=v0_spaces)
    thetaeqn0h = EmbeddedDGAdvection(state, V0h, equation_form="advective", recovered_spaces=h0_spaces)

    # build advection dictionary
    advected_fields = []
    advected_fields.append(('u', NoAdvection(state, u0, None)))
    advected_fields.append(('theta1', SSPRK3(state, theta1, thetaeqn1, limiter=ThetaLimiter(thetaeqn1))))
    advected_fields.append(('theta0v', SSPRK3(state, theta0v, thetaeqn0v, limiter=VertexBasedLimiter(VDG1))))
    advected_fields.append(('theta0h', SSPRK3(state, theta0h, thetaeqn0h, limiter=VertexBasedLimiter(VDG1))))

    # build time stepper
    stepper = AdvectionDiffusion(state, advected_fields)

    return stepper, 40.0


def run_limiters(dirname):

    stepper, tmax = setup_limiters(dirname)
    stepper.run(t=0, tmax=tmax)


def test_limiters_setup(tmpdir):

    dirname = str(tmpdir)
    run_limiters(dirname)
    filename = path.join(dirname, "limiting/diagnostics.nc")
    data = Dataset(filename, "r")

    theta1 = data.groups["theta1_perturbation"]
    max_theta1 = theta1.variables["max"]
    min_theta1 = theta1.variables["min"]

    theta0v = data.groups["theta0v_perturbation"]
    max_theta0v = theta0v.variables["max"]
    min_theta0v = theta0v.variables["min"]

    theta0h = data.groups["theta0h_perturbation"]
    max_theta0h = theta0h.variables["max"]
    min_theta0h = theta0h.variables["min"]

    assert max_theta1[-1] <= max_theta1[0]
    assert min_theta1[-1] >= min_theta1[0]
    assert max_theta0v[-1] <= max_theta0v[0]
    assert min_theta0v[-1] >= min_theta0v[0]
    assert max_theta0h[-1] <= max_theta0h[0]
    assert min_theta0h[-1] >= min_theta0h[0]
