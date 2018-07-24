from os import path
from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    Constant, SpatialCoordinate, pi, Function, sqrt, conditional, cos
from netCDF4 import Dataset
import pytest


def setup_tracer(dirname, hybridization):

    # declare grid shape, with length L and height H
    L = 1000.
    H = 1000.
    nlayers = int(H / 100.)
    ncolumns = int(L / 100.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=10.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/tracer",
                              dumpfreq=1,
                              dumplist=['u'],
                              perturbation_fields=['theta', 'rho'])
    parameters = CompressibleParameters()

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist,
                  diagnostic_fields=[Difference('theta', 'tracer')])

    # declare initial fields
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # declare tracer field and a background field
    tracer0 = state.fields("tracer", Vt)

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

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0),
                      ('tracer', tracer0)])
    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    # set up advection schemes
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = SUPGAdvection(state, Vt,
                             supg_params={"dg_direction": "horizontal"},
                             equation_form="advective")

    # build advection dictionary
    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
    advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))
    advected_fields.append(("tracer", SSPRK3(state, tracer0, thetaeqn)))

    # Set up linear solver
    if hybridization:
        linear_solver = HybridizedCompressibleSolver(state)
    else:
        linear_solver = CompressibleSolver(state)

    compressible_forcing = CompressibleForcing(state)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            compressible_forcing)

    return stepper, 100.0


def run_tracer(dirname, hybridization):

    stepper, tmax = setup_tracer(dirname, hybridization)
    stepper.run(t=0, tmax=tmax)


@pytest.mark.parametrize("hybridization", [True, False])
def test_tracer_setup(tmpdir, hybridization):

    if hybridization:
        dirname = str(tmpdir) + "_hybridization"
    else:
        dirname = str(tmpdir)

    run_tracer(dirname, hybridization)
    filename = path.join(dirname, "tracer/diagnostics.nc")
    data = Dataset(filename, "r")

    diff = data.groups["theta_minus_tracer"]
    theta = data.groups["theta"]
    diffl2 = diff["l2"][-1] / theta["l2"][0]

    assert diffl2 < 1e-5
