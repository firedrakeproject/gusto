from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, Constant, Function,
                       FunctionSpace, BrokenElement, VectorFunctionSpace)
from os import path
from netCDF4 import Dataset
import pytest

# this tests the moist-unsaturated hydrostatic balance, by setting up a vertical slice
# with this initial procedure, before taking a few time steps and ensuring that
# the resulting velocities are very small


def setup_unsaturated(dirname, recovered):

    # set up grid and time stepping parameters
    dt = 1.
    tmax = 3.
    deltax = 400
    L = 2000.
    H = 10000.

    nlayers = int(H/deltax)
    ncolumns = int(L/deltax)

    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    degree = 0 if recovered else 1

    output = OutputParameters(dirname=dirname+'/unsaturated_balance', dumpfreq=1, perturbation_fields=['water_v'])
    parameters = CompressibleParameters()
    diagnostic_fields = [Theta_d(), RelativeHumidity()]

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    if recovered:
        u_advection_option = "vector_advection_form"
    else:
        u_advection_option = "vector_invariant_form"
    eqns = MoistCompressibleEulerEquations(
        state, "CG", degree, u_advection_option=u_advection_option)

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")
    moisture = ['water_v', 'water_c']

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Isentropic background state
    Tsurf = Constant(300.)
    humidity = Constant(0.5)
    theta_d = Function(Vt).interpolate(Tsurf)
    RH = Function(Vt).interpolate(humidity)

    # Calculate hydrostatic Pi
    unsaturated_hydrostatic_balance(state, theta_d, RH)

    state.set_reference_profiles([('rho', rho0),
                                  ('theta', theta0)])

    # Set up advection schemes
    if recovered:
        VDG1 = state.spaces("DG1", "DG", 1)
        VCG1 = FunctionSpace(mesh, "CG", 1)
        Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
        Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element())
        Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

        u_opts = RecoveredOptions(embedding_space=Vu_DG1,
                                  recovered_space=Vu_CG1,
                                  broken_space=Vu,
                                  boundary_method=Boundary_Method.dynamics)
        rho_opts = RecoveredOptions(embedding_space=VDG1,
                                    recovered_space=VCG1,
                                    broken_space=Vr,
                                    boundary_method=Boundary_Method.dynamics)
        theta_opts = RecoveredOptions(embedding_space=VDG1,
                                      recovered_space=VCG1,
                                      broken_space=Vt_brok)
    else:
        rho_opts = None
        theta_opts = EmbeddedDGOptions()

    advected_fields = [SSPRK3(state, "rho", options=rho_opts),
                       SSPRK3(state, "theta", options=theta_opts),
                       SSPRK3(state, "water_v"),
                       SSPRK3(state, "water_c")]
    if recovered:
        advected_fields.append(SSPRK3(state, "u", options=u_opts))
    else:
        advected_fields.append(ImplicitMidpoint(state, "u"))

    linear_solver = CompressibleSolver(state, eqns, moisture=moisture)

    # Set up physics
    physics_list = [Condensation(state)]

    # build time stepper
    stepper = CrankNicolson(state, eqns, advected_fields,
                            linear_solver=linear_solver,
                            physics_list=physics_list)

    return stepper, tmax


def run_unsaturated(dirname, recovered):

    stepper, tmax = setup_unsaturated(dirname, recovered)
    stepper.run(t=0, tmax=tmax)


@pytest.mark.parametrize("recovered", [True, False])
def test_unsaturated_setup(tmpdir, recovered):

    dirname = str(tmpdir)
    run_unsaturated(dirname, recovered)
    filename = path.join(dirname, "unsaturated_balance/diagnostics.nc")
    data = Dataset(filename, "r")
    u = data.groups['u']
    umax = u.variables['max']

    assert umax[-1] < 1e-8
