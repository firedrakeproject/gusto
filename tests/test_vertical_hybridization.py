from gusto import *
from firedrake import *


def run_incompressible_balance_test(dirname):
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
    L = 1.e5
    m = PeriodicRectangleMesh(columns, 1, L, 1.e4, quadrilateral=True)
    dt = 6.0

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    fieldlist = ['u', 'p', 'b']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/test_incompressible", dumplist=['u'], dumpfreq=5)
    parameters = CompressibleParameters()

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="RTCF",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist)

    b0 = state.fields("b")

    # z.grad(bref) = N**2
    x, y, z = SpatialCoordinate(mesh)
    N = parameters.N
    bref = z*(N**2)
    b_b = Function(b0.function_space()).interpolate(bref)
    b0.interpolate(b_b)

    # get F
    Vv = state.spaces("Vv")
    v = TrialFunction(Vv)
    w = TestFunction(Vv)
    bcs = [DirichletBC(Vv, 0.0, "bottom")]

    af = inner(w, v)*dx
    Lf = inner(state.k, w)*b0*dx
    F = Function(Vv)

    solve(af == Lf, F, bcs=bcs)

    # Define mixed function space
    VDG = state.spaces("DG")
    WV = Vv*VDG

    # Set up mixed problem
    v, pprime = TrialFunctions(WV)
    w, phi = TestFunctions(WV)

    bcs = [DirichletBC(WV[0], 0.0, "bottom")]

    a = (
        inner(w, v) + div(w)*pprime + div(v)*phi
    )*dx
    L = phi*div(F)*dx
    w_hybrid = Function(WV)
    w_ref = Function(WV)

    hybrid_params = {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'mat_type': 'matfree',
        'pc_python_type': 'gusto.VerticalHybridizationPC',
        'vert_hybridization': {'ksp_type': 'preonly',
                               'pc_type': 'lu',
                               'pc_factor_mat_solver_type': 'mumps'}
    }
    solve(a == L, w_hybrid, bcs=bcs, solver_parameters=hybrid_params)
    _, pprime_hybrid = w_hybrid.split()

    ref_params = {'ksp_type': 'preonly',
                  'mat_type': 'aij',
                  'pc_type': 'lu',
                  'pc_factor_mat_solver_type': 'mumps'}
    solve(a == L, w_ref, bcs=bcs, solver_parameters=ref_params)
    _, pprime_ref = w_ref.split()

    return errornorm(pprime_ref, pprime_hybrid, norm_type="L2")


def run_compressible_balance_test(dirname):
    dt = 1.
    deltax = 400
    L = 2000.
    H = 10000.

    nlayers = int(H/deltax)
    ncolumns = int(L/deltax)

    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+'/test_compressible', dumpfreq=10, dumplist=['u'])
    parameters = CompressibleParameters()
    diagnostics = Diagnostics(*fieldlist)
    diagnostic_fields = []

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  fieldlist=fieldlist,
                  diagnostic_fields=diagnostic_fields)

    theta0 = state.fields("theta")

    # Isentropic background state
    Tsurf = Constant(300.)
    theta0.interpolate(Tsurf)

    # Calculate hydrostatic Pi
    VDG = state.spaces("DG")
    Vv = state.spaces("Vv")
    W = MixedFunctionSpace((Vv, VDG))
    v, pi = TrialFunctions(W)
    dv, dpi = TestFunctions(W)

    n = FacetNormal(state.mesh)
    cp = state.parameters.cp

    a = (
        (cp*inner(v, dv) - cp*div(dv*theta0)*pi)*dx
        + dpi*div(theta0*v)*dx
    )

    L = -cp*inner(dv, n)*theta0*ds_b
    g = state.parameters.g
    L -= g*inner(dv, state.k)*dx

    bcs = [DirichletBC(W.sub(0), 0.0, "top")]
    w_hybrid = Function(W)
    w_ref = Function(W)

    hybrid_params = {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'mat_type': 'matfree',
        'pc_python_type': 'gusto.VerticalHybridizationPC',
        'vert_hybridization': {'ksp_type': 'preonly',
                               'pc_type': 'lu',
                               'pc_factor_mat_solver_type': 'mumps'}
    }
    solve(a == L, w_hybrid, bcs=bcs, solver_parameters=hybrid_params)
    _, pi_hybrid = w_hybrid.split()

    ref_params = {'ksp_type': 'preonly',
                  'mat_type': 'aij',
                  'pc_type': 'lu',
                  'pc_factor_mat_solver_type': 'mumps'}
    solve(a == L, w_ref, bcs=bcs, solver_parameters=ref_params)
    _, pi_ref = w_ref.split()

    return errornorm(pi_ref, pi_hybrid, norm_type="L2")


# def test_incompressible_balance_hybrid(tmpdir):
#     """Solves the incompressible hydrostatic equation
#     for a pressure variable using two solver configurations:

#     (1): Hybridization with LU on the multiplier solve;
#     (2): Direct LU on the full system.

#     The error between the two fields should be small.
#     """

#     dirname = str(tmpdir)
#     error = run_incompressible_balance_test(dirname)

#     assert error < 1.0e-8


def test_compressible_balance_hybrid(tmpdir):
    """Solves the compressible hydrostatic equation
    for a pressure variable using two solver configurations:

    (1): Hybridization with LU on the multiplier solve;
    (2): Direct LU on the full system.

    The error between the two fields should be small.
    """

    dirname = str(tmpdir)
    error = run_compressible_balance_test(dirname)

    assert error < 1.0e-9
