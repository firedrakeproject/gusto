from os import path
from gusto import *
from firedrake import Constant, SpatialCoordinate, pi, Function, sqrt, conditional, cos
import json


def setup_tracer(dirname):

    # declare grid shape, with length L and height H
    L = 1000.
    H = 1000.
    nlayers = int(H / 100.)
    ncolumns = int(L / 100.)
    physical_domain = VerticalSlice(H=H, L=L, ncolumns=ncolumns, nlayers=nlayers)
    timestepping = TimesteppingParameters(dt=10.0)
    output = OutputParameters(dirname=dirname+"/tracer",
                              dumpfreq=1,
                              dumplist=['u'],
                              perturbation_fields=['theta', 'rho'])

    state = CompressibleEulerState(physical_domain.mesh,
                                   output=output,
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

    # set up perturbation to theta
    xc = 500.
    zc = 350.
    rc = 250.
    x = SpatialCoordinate(physical_domain.mesh)
    r = sqrt((x[0]-xc)**2 + (x[1]-zc)**2)
    theta_pert = conditional(r > rc, 0., 0.25*(1. + cos((pi/rc)*r)))

    theta0.interpolate(theta_b + theta_pert)
    tracer0.interpolate(theta0)

    eqn = SUPGAdvection(physical_domain, Vt, Vu,
                        dt=timestepping.dt,
                        supg_params={"dg_direction": "horizontal"},
                        equation_form="advective")
    advected_fields = [(("tracer", SSPRK3(tracer0, timestepping.dt, eqn)))]

    model = CompressibleEulerModel(state,
                                   physical_domain,
                                   is_rotating=False,
                                   timestepping=timestepping,
                                   advected_fields=advected_fields)

    rho_b = Function(Vr)

    # Calculate initial rho
    compressible_hydrostatic_balance(state, model.parameters,
                                     physical_domain.vertical_normal,
                                     theta_b, rho_b,
                                     solve_for_rho=True)
    rho0.interpolate(rho_b)

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0),
                      ('tracer', tracer0)])
    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    # build time stepper
    stepper = Timestepper(model)

    return stepper, 100.0


def run_tracer(dirname):

    stepper, tmax = setup_tracer(dirname)
    stepper.run(t=0, tmax=tmax)


def test_tracer_setup(tmpdir):

    dirname = str(tmpdir)
    run_tracer(dirname)
    with open(path.join(dirname, "tracer/diagnostics.json"), "r") as f:
        data = json.load(f)

    diffl2 = data["theta_minus_tracer"]["l2"][-1] / data["theta"]["l2"][0]

    assert diffl2 < 1e-5
