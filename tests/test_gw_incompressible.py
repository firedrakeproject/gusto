from gusto import *
from firedrake import Expression, PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, sin
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

    # vertical normal
    k = Constant([0, 1])

    fieldlist = ['u', 'p', 'b']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/gw_incompressible", dumplist=['u'], dumpfreq=5)
    diagnostics = Diagnostics(*fieldlist)
    parameters = CompressibleParameters()
    diagnostic_fields = [CourantNumber()]

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  vertical_normal=k,
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  fieldlist=fieldlist,
                  diagnostic_fields=diagnostic_fields)

    # Initial conditions
    u0, p0, b0 = Function(state.V[0]), Function(state.V[1]), Function(state.V[2])

    # z.grad(bref) = N**2
    N = parameters.N
    x, z = SpatialCoordinate(mesh)
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
    state.set_reference_profiles({'b':b_b})
    state.output.meanfields = ['b']

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

    return stepper, 10*dt


def run_gw_incompressible(dirname):

    stepper, tmax = setup_gw(dirname)
    stepper.run(t=0, tmax=tmax)


def test_gw(tmpdir):

    dirname = str(tmpdir)
    run_gw_incompressible(dirname)
