"""
This tests the moist saturated hydrostatic balance, by setting up a vertical
slice with the appropriate initialisation procedure, before taking a few time
steps and ensuring that the resulting velocities are very small.
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, Constant, Function,
                       FunctionSpace, VectorFunctionSpace)
from os import path
from netCDF4 import Dataset
import pytest


def setup_saturated(dirname, recovered):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Parameters
    dt = 1.
    tmax = 3.
    deltax = 400.
    L = 2000.
    H = 10000.
    nlayers = int(H/deltax)
    ncolumns = int(L/deltax)
    degree = 0 if recovered else 1

    # Domain
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "CG", degree)

    # Equation
    tracers = [WaterVapour(), CloudWater()]
    if recovered:
        u_transport_option = "vector_advection_form"
    else:
        u_transport_option = "vector_invariant_form"
    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(
        domain, parameters, u_transport_option=u_transport_option, active_tracers=tracers)

    # I/O
    output = OutputParameters(dirname=dirname+'/saturated_balance', dumpfreq=1, dumplist=['u'])
    diagnostic_fields = [Theta_e(eqns)]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Set up transport schemes
    if recovered:
        VDG1 = domain.spaces("DG1_equispaced")
        VCG1 = FunctionSpace(mesh, "CG", 1)
        Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element())
        Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

        u_opts = RecoveryOptions(embedding_space=Vu_DG1,
                                 recovered_space=Vu_CG1,
                                 boundary_method=BoundaryMethod.taylor)
        rho_opts = RecoveryOptions(embedding_space=VDG1,
                                   recovered_space=VCG1,
                                   boundary_method=BoundaryMethod.taylor)
        theta_opts = RecoveryOptions(embedding_space=VDG1,
                                     recovered_space=VCG1)
        wv_opts = RecoveryOptions(embedding_space=VDG1,
                                  recovered_space=VCG1)
        wc_opts = RecoveryOptions(embedding_space=VDG1,
                                  recovered_space=VCG1)
    else:

        rho_opts = None
        theta_opts = EmbeddedDGOptions()
        wv_opts = EmbeddedDGOptions()
        wc_opts = EmbeddedDGOptions()

    transported_fields = [SSPRK3(domain, 'rho', options=rho_opts),
                          SSPRK3(domain, 'theta', options=theta_opts),
                          SSPRK3(domain, 'water_vapour', options=wv_opts),
                          SSPRK3(domain, 'cloud_water', options=wc_opts)]

    if recovered:
        transported_fields.append(SSPRK3(domain, 'u', options=u_opts))
    else:
        transported_fields.append(TrapeziumRule(domain, 'u'))

    transport_methods = [DGUpwind(eqns, 'u'),
                         DGUpwind(eqns, 'rho'),
                         DGUpwind(eqns, 'theta'),
                         DGUpwind(eqns, 'water_vapour'),
                         DGUpwind(eqns, 'cloud_water')]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Physics schemes
    physics_schemes = [(SaturationAdjustment(eqns), ForwardEuler(domain))]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                      transport_methods,
                                      linear_solver=linear_solver,
                                      physics_schemes=physics_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")
    water_v0 = stepper.fields("water_vapour")
    water_c0 = stepper.fields("cloud_water")

    # spaces
    Vt = theta0.function_space()

    # Isentropic background state
    Tsurf = Constant(300.)
    total_water = Constant(0.02)
    theta_e = Function(Vt).interpolate(Tsurf)
    water_t = Function(Vt).interpolate(total_water)

    # Calculate hydrostatic exner
    saturated_hydrostatic_balance(eqns, stepper.fields, theta_e, water_t)
    water_c0.assign(water_t - water_v0)

    stepper.set_reference_profiles([('rho', rho0), ('theta', theta0)])

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
