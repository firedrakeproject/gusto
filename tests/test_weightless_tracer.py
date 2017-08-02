from os import path
from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    Constant, SpatialCoordinate, pi, Function, sqrt, conditional, cos
import json


def setup_tracer(dirname):

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
    schur_params = {'pc_type': 'fieldsplit',
                    'pc_fieldsplit_type': 'schur',
                    'ksp_type': 'gmres',
                    'ksp_monitor_true_residual': True,
                    'ksp_max_it': 100,
                    'ksp_gmres_restart': 50,
                    'pc_fieldsplit_schur_fact_type': 'FULL',
                    'pc_fieldsplit_schur_precondition': 'selfp',
                    'fieldsplit_0_ksp_type': 'richardson',
                    'fieldsplit_0_ksp_max_it': 5,
                    'fieldsplit_0_pc_type': 'bjacobi',
                    'fieldsplit_0_sub_pc_type': 'ilu',
                    'fieldsplit_1_ksp_type': 'richardson',
                    'fieldsplit_1_ksp_max_it': 5,
                    'fieldsplit_1_ksp_monitor_true_residual': True,
                    'fieldsplit_1_pc_type': 'gamg',
                    'fieldsplit_1_pc_gamg_sym_graph': True,
                    'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                    'fieldsplit_1_mg_levels_ksp_chebyshev_esteig': True,
                    'fieldsplit_1_mg_levels_ksp_chebyshev_esteig_random': True,
                    'fieldsplit_1_mg_levels_ksp_max_it': 5,
                    'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                    'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

    linear_solver = CompressibleSolver(state, params=schur_params)

    compressible_forcing = CompressibleForcing(state)

    # build time stepper
    stepper = Timestepper(state, advected_fields, linear_solver,
                          compressible_forcing)

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
