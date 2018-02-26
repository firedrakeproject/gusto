from os import path
from gusto import *
from firedrake import as_vector, Constant, PeriodicIntervalMesh, \
    SpatialCoordinate, ExtrudedMesh, FunctionSpace, Function, \
    conditional, sqrt, FiniteElement, TensorProductElement, BrokenElement
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from netCDF4 import Dataset

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then advected by a prescribed advection scheme


def setup_vert_limiters(dirname):

    # declare grid shape
    L = 200.
    H = 800.
    ncolumns = int(L / 20.)
    nlayers = int(H / 20.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x, z = SpatialCoordinate(mesh)

    fieldlist = ['u']
    timestepping = TimesteppingParameters(dt=1.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/limiting_vert",
                              dumpfreq=5,
                              dumplist=['u'],
                              perturbation_fields=['theta0', 'theta1'])
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
    CG2_element = FiniteElement("CG", cell, 2)
    V0_element = TensorProductElement(DG0_element, CG1_element)
    V1_element = TensorProductElement(DG1_element, CG2_element)

    # spaces
    VDG1 = FunctionSpace(mesh, "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    V0 = FunctionSpace(mesh, V0_element)
    V1 = FunctionSpace(mesh, V1_element)

    V0_brok = FunctionSpace(mesh, BrokenElement(V0.ufl_element()))

    V0_spaces = (VDG1, VCG1, V0_brok)

    # declare initial fields
    u0 = state.fields("u")
    theta0 = state.fields("theta0", V0)
    theta1 = state.fields("theta1", V1)

    # Isentropic background state
    Tsurf = 0.
    thetab = Constant(Tsurf)
    theta_b1 = Function(V1).interpolate(thetab)
    theta_b0 = Function(V0).interpolate(thetab)

    # set up bubble
    xc = L / 2
    zc = 700.
    rc = 80.
    theta_expr = conditional(sqrt((x - xc) ** 2.0) < rc,
                             conditional(sqrt((z - zc) ** 2.0) < rc,
                                         Constant(2.0),
                                         Constant(0.0)), Constant(0.0))
    theta_pert1 = Function(V1).interpolate(theta_expr)
    theta_pert0 = Function(V0).interpolate(theta_expr)

    # set up velocity field
    u_max = Constant(5.0)
    u0.project(as_vector([0, -u_max]))
    theta0.interpolate(theta_b0 + theta_pert0)
    theta1.interpolate(theta_b1 + theta_pert1)

    state.initialise([('u', u0), ('theta1', theta1), ('theta0', theta0)])
    state.set_reference_profiles([('theta1', theta_b1), ('theta0', theta_b0)])

    # set up advection schemes
    thetaeqn1 = EmbeddedDGAdvection(state, V1, equation_form="advective")
    thetaeqn0 = EmbeddedDGAdvection(state, V0, equation_form="advective", recovered_spaces=V0_spaces)

    # build advection dictionary
    advected_fields = []
    advected_fields.append(('u', NoAdvection(state, u0, None)))
    advected_fields.append(('theta1', SSPRK3(state, theta1, thetaeqn1, limiter=ThetaLimiter(thetaeqn1))))
    advected_fields.append(('theta0', SSPRK3(state, theta0, thetaeqn0, limiter=VertexBasedLimiter(VDG1))))

    # build time stepper
    stepper = AdvectionDiffusion(state, advected_fields)

    return stepper, 40.0


def run_vert_limiters(dirname):

    stepper, tmax = setup_vert_limiters(dirname)
    stepper.run(t=0, tmax=tmax)


def test_vert_limiters_setup(tmpdir):

    dirname = str(tmpdir)
    run_vert_limiters(dirname)
    filename = path.join(dirname, "limiting_vert/diagnostics.nc")
    data = Dataset(filename, "r")

    theta1 = data.groups["theta1_perturbation"]
    max_theta1 = theta1.variables["max"]
    min_theta1 = theta1.variables["min"]

    theta0 = data.groups["theta0_perturbation"]
    max_theta0 = theta0.variables["max"]
    min_theta0 = theta0.variables["min"]

    assert max_theta1[-1] <= max_theta1[0]
    assert min_theta1[-1] >= min_theta1[0]
    assert max_theta0[-1] <= max_theta0[0]
    assert min_theta0[-1] >= min_theta0[0]
