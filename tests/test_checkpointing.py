from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, exp, sin, Function, as_vector)
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

    output = OutputParameters(dirname=dirname+"/sk_nonlinear", dumplist=['u'], dumpfreq=5, log_level=INFO)
    parameters = CompressibleParameters()
    diagnostic_fields = [CourantNumber()]

    state = State(mesh, dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    eqns = CompressibleEulerEquations(state, "CG", 1, 1)

    # Initial conditions
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
    compressible_hydrostatic_balance(state, theta_b, rho_b)

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

    # Set up advection schemes
    advected_fields = []
    advected_fields.append(ImplicitMidpoint(state, eqns, advection, field_name="u"))
    advected_fields.append(SSPRK3(state, eqns, advection, field_name="rho"))
    advected_fields.append(SSPRK3(state, eqns, advection, field_name="theta"))

    # build time stepper
    stepper = CrankNicolson(state, equation_set=eqns, advected_fields=advected_fields)

    return stepper, 2*dt


def test_checkpointing(tmpdir):

    dirname = str(tmpdir)
    stepper, tmax = setup_sk(dirname)
    stepper.run(t=0., tmax=tmax)
    dt = stepper.state.dt
    stepper.run(t=0, tmax=2*tmax+dt, pickup=True)
