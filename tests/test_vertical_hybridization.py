from gusto import *
from firedrake import *


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
    v_hybrid, pi_hybrid = w_hybrid.split()

    ref_params = {'ksp_type': 'preonly',
                  'mat_type': 'aij',
                  'pc_type': 'lu',
                  'pc_factor_mat_solver_type': 'mumps'}
    solve(a == L, w_ref, bcs=bcs, solver_parameters=ref_params)
    v_ref, pi_ref = w_ref.split()

    v_error = errornorm(v_ref, v_hybrid, norm_type="L2")
    pi_error = errornorm(pi_ref, pi_hybrid, norm_type="L2")

    return v_error, pi_error


def test_compressible_balance_hybrid(tmpdir):
    """Solves the compressible hydrostatic equation
    for a pressure variable using two solver configurations:

    (1): Hybridization with LU on the multiplier solve;
    (2): Direct LU on the full system.

    The error between the two fields should be small.
    """

    dirname = str(tmpdir)
    v_error, pi_error = run_compressible_balance_test(dirname)

    assert v_error < 1.0e-9
    assert pi_error < 1.0e-9
