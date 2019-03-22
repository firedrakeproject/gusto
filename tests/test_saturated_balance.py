from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, Constant, Function,
                       FunctionSpace, BrokenElement, VectorFunctionSpace)
from os import path
from netCDF4 import Dataset

# this tests the moist-saturated hydrostatic balance, by setting up a vertical slice
# with this initial procedure, before taking a few time steps and ensuring that
# the resulting velocities are very small


def setup_saturated(dirname):

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
    recovered = True
    degree = 0 if recovered else 1

    output = OutputParameters(dirname=dirname+'/saturated_balance', dumpfreq=1, dumplist=['u'], perturbation_fields=['water_v'])
    parameters = CompressibleParameters()
    diagnostic_fields = [Theta_e()]

    state = State(mesh, dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    eqns = MoistCompressibleEulerEquations(state, family="CG",
                                           horizontal_degree=degree,
                                           vertical_degree=degree)

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")
    water_v0 = state.fields("water_v", theta0.function_space())
    water_c0 = state.fields("water_c", theta0.function_space())

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

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0),
                      ('water_v', water_v0),
                      ('water_c', water_c0)])
    state.set_reference_profiles([('rho', rho0),
                                  ('theta', theta0),
                                  ('water_v', water_v0)])

    # Set up advection schemes
    if recovered:
        VDG1 = FunctionSpace(mesh, "DG", 1)
        VCG1 = FunctionSpace(mesh, "CG", 1)
        Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
        Vu_DG1 = VectorFunctionSpace(mesh, "DG", 1)
        Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

        u_opts = RecoveredOptions(embedding_space=Vu_DG1,
                                  recovered_space=Vu_CG1,
                                  broken_space=Vu)
        rho_opts = RecoveredOptions(embedding_space=VDG1,
                                    recovered_space=VCG1,
                                    broken_space=Vr)
        theta_opts = RecoveredOptions(embedding_space=VDG1,
                                      recovered_space=VCG1,
                                      broken_space=Vt_brok)
        advected_fields = [SSPRK3(state, eqns, advection, field_name="u",
                                  options=u_opts),
                           SSPRK3(state, eqns, advection, field_name="rho",
                                  options=rho_opts),
                           SSPRK3(state, eqns, advection, field_name="theta",
                                  options=theta_opts)]
    else:
        advected_fields = [ImplicitMidpoint(state, eqns, advection,
                                            field_name="u"),
                           SSPRK3(state, eqns, advection, field_name="rho"),
                           SSPRK3(state, eqns, advection, field_name="theta")]

    advected_fields.append(SSPRK3(state, eqns, advection, field_name="water_v"))
    advected_fields.append(SSPRK3(state, eqns, advection, field_name="water_c"))

    # add physics
    physics_list = [Condensation(state)]

    # build time stepper
    stepper = CrankNicolson(state, equation_set=eqns,
                            schemes=advected_fields, physics_list=physics_list)

    return stepper, tmax


def run_saturated(dirname):

    stepper, tmax = setup_saturated(dirname)
    stepper.run(t=0, tmax=tmax)


def test_saturated_setup(tmpdir):

    dirname = str(tmpdir)
    run_saturated(dirname)
    filename = path.join(dirname, "saturated_balance/diagnostics.nc")
    data = Dataset(filename, "r")
    u = data.groups['u']
    umax = u.variables['max']

    assert umax[-1] < 1e-5
