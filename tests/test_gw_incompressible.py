from gusto import *
from firedrake import SpatialCoordinate, \
    VectorFunctionSpace, PeriodicRectangleMesh, ExtrudedMesh, \
    sin, pi


def max(f):
    fmax = op2.Global(1, [-1000], dtype=float)
    op2.par_loop(op2.Kernel("""void maxify(double *a, double *b)
    {
    a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
    }""", "maxify"),
                 f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]


def setup_gw(dirname):
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
    L = 1.e5
    m = PeriodicRectangleMesh(columns, 1, L, 1.e4, quadrilateral=True)
    dt = 6.0

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    # Space for initialising velocity
    W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)

    # vertical coordinate and normal
    x, y, z = SpatialCoordinate(mesh)
    k = Constant([0, 0, 1])

    fieldlist = ['u', 'p', 'b']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/gw_incompressible", dumplist=['u'], dumpfreq=5)
    diagnostics = Diagnostics(*fieldlist)
    parameters = CompressibleParameters(geopotential=False)
    diagnostic_fields = [CourantNumber()]

    state = IncompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                                family="RTCF",
                                z=z, k=k,
                                timestepping=timestepping,
                                output=output,
                                parameters=parameters,
                                diagnostics=diagnostics,
                                fieldlist=fieldlist,
                                diagnostic_fields=diagnostic_fields,
                                on_sphere=False)

    # Initial conditions
    u0, p0, b0 = Function(state.V[0]), Function(state.V[1]), Function(state.V[2])

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    N = parameters.N

    # z.grad(bref) = N**2
    N = parameters.N
    bref = z*(N**2)

    b_b = Function(state.V[2]).interpolate(bref)

    a = Constant(5.0e3)
    deltab = Constant(1.0e-2)
    H = Constant(H)
    L = Constant(L)
    b_pert = deltab*sin(pi*z/H)/(1 + (x - L/2)**2/a**2)
    b0.interpolate(b_b + b_pert)
    incompressible_hydrostatic_balance(state, b0, p0)
    uinit = Function(W_VectorCG1).interpolate(as_vector([20.0,0.0,0.0]))
    u0.project(uinit)

    state.initialise([u0, p0, b0])
    state.set_reference_profiles(b_b)
    state.output.meanfields = {'b':state.bbar}

    # Set up advection schemes
    ueqn = EulerPoincare(state, state.V[0])
    beqn = EmbeddedDGAdvection(state, state.V[2], equation_form="advective")
    advection_dict = {}
    advection_dict["u"] = ThetaMethod(state, u0, ueqn)
    advection_dict["b"] = SSPRK3(state, b0, beqn)

    # Set up linear solver
    params = {'ksp_type':'gmres',
              'pc_type':'fieldsplit',
              'pc_fieldsplit_type':'additive',
              'fieldsplit_0_pc_type':'lu',
              'fieldsplit_1_pc_type':'lu',
              'fieldsplit_0_ksp_type':'preonly',
              'fieldsplit_1_ksp_type':'preonly'}
    linear_solver = IncompressibleSolver(state, L, params=params)

    # Set up forcing
    forcing = IncompressibleForcing(state)

    # build time stepper
    stepper = Timestepper(state, advection_dict, linear_solver,
                          forcing)

    return b0, p0, mesh, state, stepper, 5*dt


def run_gw_incompressible(dirname):

    b0, p0, mesh, state, stepper, tmax = setup_gw(dirname)
    stepper.run(t=0, tmax=tmax)

    # get pressure gradient
    V0 = state.V[0]
    g = TrialFunction(V0)
    wg = TestFunction(V0)

    n = FacetNormal(mesh)

    a = inner(wg,g)*dx
    L = -div(wg)*p0*dx + inner(wg,n)*p0*ds_tb
    pgrad = Function(V0)
    solve(a == L, pgrad)

    # get difference between b0 and dp0/dz
    V1 = state.V[1]
    phi = TestFunction(V1)
    m = TrialFunction(V1)

    a = phi*m*dx
    L = phi*(b0-pgrad[2])*dx
    diff = Function(V1)
    solve(a == L, diff)

    # get v component of the velocity
    u = state.field_dict['u']
    v = u[1]

    tri = TrialFunction(V1)
    tes = TestFunction(V1)

    ax = inner(tes,tri)*dx
    Lx = inner(tes,v)*dx
    vproj = Function(V1)
    solve(ax == Lx, vproj)

    return diff, vproj


def test_gw(tmpdir):

    dirname = str(tmpdir)
    diff, v = run_gw_incompressible(dirname)
    assert max(diff) < 0.05
    assert max(v) < 1.e-10
