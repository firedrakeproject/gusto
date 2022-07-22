"""
This tests the moist saturated hydrostatic balance, by setting up a vertical
slice with the appropriate initialisation procedure, before taking a few time
steps and ensuring that the resulting velocities are very small.
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, Constant, Function,
                       FunctionSpace, BrokenElement, VectorFunctionSpace)
from os import path
from netCDF4 import Dataset
import pytest


def setup_saturated(dirname, recovered):

    # set up grid and time stepping parameters
    dt = 1.
    tmax = 3.
    deltax = 400.
    L = 2000.
    H = 10000.

    nlayers = int(H/deltax)
    ncolumns = int(L/deltax)

    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    # option to easily change between recovered and not if necessary
    # default should be to use lowest order set of spaces
    degree = 0 if recovered else 1

    output = OutputParameters(dirname=dirname+'/saturated_balance', dumpfreq=1, dumplist=['u'])
    parameters = CompressibleParameters()
    diagnostic_fields = [Theta_e()]

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    tracers = [WaterVapour(), CloudWater()]

    if recovered:
        u_transport_option = "vector_advection_form"
    else:
        u_transport_option = "vector_invariant_form"
    eqns = CompressibleEulerEquations(
        state, "CG", degree, u_transport_option=u_transport_option, active_tracers=tracers)

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")
    water_v0 = state.fields("vapour_mixing_ratio")
    water_c0 = state.fields("cloud_liquid_mixing_ratio")
    moisture = ['vapour_mixing_ratio', 'cloud_liquid_mixing_ratio']

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Isentropic background state
    Tsurf = Constant(300.)
    total_water = Constant(0.02)
    theta_e = Function(Vt).interpolate(Tsurf)
    water_t = Function(Vt).interpolate(total_water)

    # Calculate hydrostatic Pi
    saturated_hydrostatic_balance(state, theta_e, water_t)
    water_c0.assign(water_t - water_v0)

    state.set_reference_profiles([('rho', rho0),
                                  ('theta', theta0)])

    # Set up transport schemes
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
        wv_opts = RecoveredOptions(embedding_space=VDG1,
                                   recovered_space=VCG1,
                                   broken_space=Vt_brok)
        wc_opts = RecoveredOptions(embedding_space=VDG1,
                                   recovered_space=VCG1,
                                   broken_space=Vt_brok)
    else:

        rho_opts = None
        theta_opts = EmbeddedDGOptions()
        wv_opts = EmbeddedDGOptions()
        wc_opts = EmbeddedDGOptions()

    transported_fields = [SSPRK3(state, 'rho', options=rho_opts),
                          SSPRK3(state, 'theta', options=theta_opts),
                          SSPRK3(state, 'vapour_mixing_ratio', options=wv_opts),
                          SSPRK3(state, 'cloud_liquid_mixing_ratio', options=wc_opts)]

    if recovered:
        transported_fields.append(SSPRK3(state, 'u', options=u_opts))
    else:
        transported_fields.append(ImplicitMidpoint(state, 'u'))

    linear_solver = CompressibleSolver(state, eqns, moisture=moisture)

    # add physics
    physics_list = [Condensation(state)]

    # build time stepper
    stepper = CrankNicolson(state, eqns, transported_fields,
                            linear_solver=linear_solver,
                            physics_list=physics_list)

    return stepper, tmax


def run_saturated(dirname, recovered):

    stepper, tmax = setup_saturated(dirname, recovered)
    stepper.run(t=0, tmax=tmax)


@pytest.mark.parametrize("recovered", [True, False])
def test_saturated_setup(tmpdir, recovered):

    dirname = str(tmpdir)
    run_saturated(dirname, recovered)
    filename = path.join(dirname, "saturated_balance/diagnostics.nc")
    data = Dataset(filename, "r")
    u = data.groups['u']
    umax = u.variables['max']

    assert umax[-1] < 1e-5
