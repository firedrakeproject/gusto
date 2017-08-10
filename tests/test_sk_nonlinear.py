from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, exp, sin, Function, as_vector
import numpy as np


def setup_sk(dirname):
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
    L = 1.e5
    m = PeriodicIntervalMesh(columns, L)
    dt = 6.0

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/sk_nonlinear", dumplist=['u'], dumpfreq=5, Verbose=True)

    model = CompressibleEulerModel(mesh,
                                   timestepping=timestepping,
                                   output=output)

    # Initial conditions
    state = model.state
    parameters = model.parameters
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = parameters.g
    N = parameters.N

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    x, z = SpatialCoordinate(mesh)
    Tsurf = 300.
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic Pi
    compressible_hydrostatic_balance(state, parameters, theta_b, rho_b)

    a = 5.0e3
    deltaTheta = 1.0e-2
    theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([20.0, 0.0]))

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0)])
    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    model.setup()
    # build time stepper
    stepper = Timestepper(model)

    return stepper, 10*dt


def run_sk(dirname):

    stepper, tmax = setup_sk(dirname)
    stepper.run(t=0., tmax=tmax)


def test_sk(tmpdir):

    dirname = str(tmpdir)
    run_sk(dirname)
