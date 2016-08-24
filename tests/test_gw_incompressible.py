from gusto import *
from firedrake import Expression, \
    VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh, \
    sin
import numpy as np


def setup_gw(dirname):
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
    L = 1.e5
    m = PeriodicIntervalMesh(columns, L)
    dt = 6.0

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    # Space for initialising velocity
    W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
    W_CG1 = FunctionSpace(mesh, "CG", 1)

    # vertical coordinate and normal
    z = Function(W_CG1).interpolate(Expression("x[1]"))
    k = Function(W_VectorCG1).interpolate(Expression(("0.","1.")))

    fieldlist = ['u', 'p', 'b']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/gw_incompressible", dumplist=['u'], dumpfreq=5)
    diagnostics = Diagnostics(*fieldlist)
    parameters = CompressibleParameters(geopotential=False)
    diagnostic_fields = [CourantNumber()]

    state = IncompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                                family="CG",
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

    W_DG1 = FunctionSpace(mesh, "DG", 1)
    x = Function(W_DG1).interpolate(Expression("x[0]"))
    a = 5.0e3
    deltaTheta = 1.0e-2
    theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
    b0.interpolate(b_b + theta_pert)
    u0.project(as_vector([20.0,0.0]))

    state.initialise([u0, p0, b0])
    state.set_reference_profiles(b_b)
    state.output.meanfields = {'b':state.bbar}

    # Set up advection schemes
    Vtdg = FunctionSpace(mesh, "DG", 1)
    advection_dict = {}
    advection_dict["u"] = EulerPoincareForm(state, state.V[0])
    advection_dict["b"] = EmbeddedDGAdvection(state, state.V[2],
                                              Vdg=Vtdg, continuity=False)

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

    return stepper, 10*dt


def run_gw_incompressible(dirname):

    stepper, tmax = setup_gw(dirname)
    stepper.run(t=0, tmax=tmax)


def test_gw(tmpdir):

    dirname = str(tmpdir)
    run_gw_incompressible(dirname)
