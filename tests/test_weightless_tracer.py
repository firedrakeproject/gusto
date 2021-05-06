from os import path
from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       Constant, SpatialCoordinate, pi, Function,
                       sqrt, conditional, cos)
from netCDF4 import Dataset


def setup_tracer(dirname):

    # declare grid shape, with length L and height H
    L = 1000.
    H = 1000.
    nlayers = int(H / 100.)
    ncolumns = int(L / 100.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))

    dt = 10.
    output = OutputParameters(dirname=dirname+"/tracer",
                              dumpfreq=1,
                              dumplist=['u'],
                              perturbation_fields=['theta', 'rho'])
    parameters = CompressibleParameters()

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=[Difference('theta', 'tracer')])

    eqns = CompressibleEulerEquations(state, "CG", 1)

    # Initial density and potential temperature fields
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    tracer_eqn = AdvectionEquation(state, theta0.function_space(), "tracer")
    # Initial tracer
    tracer0 = state.fields("tracer")

    # spaces
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Isentropic background state
    Tsurf = Constant(300.)

    theta_b = Function(Vt).interpolate(Tsurf)
    rho_b = Function(Vr)

    # Calculate initial rho
    compressible_hydrostatic_balance(state, theta_b, rho_b,
                                     solve_for_rho=True)

    # set up perturbation to theta
    xc = 500.
    zc = 350.
    rc = 250.
    x = SpatialCoordinate(mesh)
    r = sqrt((x[0]-xc)**2 + (x[1]-zc)**2)
    theta_pert = conditional(r > rc, 0., 0.25*(1. + cos((pi/rc)*r)))

    theta0.interpolate(theta_b + theta_pert)
    rho0.interpolate(rho_b)
    tracer0.interpolate(theta0)

    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    # set up advection schemes
    advection_schemes = [ImplicitMidpoint(state, "u"),
                         SSPRK3(state, "rho"),
                         SSPRK3(state, "theta", options=SUPGOptions())]

    # Set up linear solver
    linear_solver = CompressibleSolver(state, eqns)

    # build time stepper
    stepper = CrankNicolson(state, eqns, advection_schemes,
                            auxiliary_equations_and_schemes=[(tracer_eqn, SSPRK3(state, options=SUPGOptions()))],
                            linear_solver=linear_solver)

    return stepper, 100.0


def run_tracer(dirname):

    stepper, tmax = setup_tracer(dirname)
    stepper.run(t=0, tmax=tmax)


def test_tracer_setup(tmpdir):

    dirname = str(tmpdir)

    run_tracer(dirname)
    filename = path.join(dirname, "tracer/diagnostics.nc")
    data = Dataset(filename, "r")

    diff = data.groups["theta_minus_tracer"]
    theta = data.groups["theta"]
    diffl2 = diff["l2"][-1] / theta["l2"][0]

    assert diffl2 < 1e-5
