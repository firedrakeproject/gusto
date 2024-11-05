"""
Tests the CoupledTransportEquation class.
Two tracers are transported using combinations of advective and
conservative forms. The tracers are set to be a mixing ratio when
using the advective form and a density for the conservative form.

"""

from firedrake import norm, BrokenElement
from gusto import *
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    norm1 = norm(timestepper.fields("f1") - f_end) / norm(f_end)
    norm2 = norm(timestepper.fields("f2") - f_end) / norm(f_end)
    return norm1, norm2


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form1", ["advective", "continuity"])
@pytest.mark.parametrize("equation_form2", ["advective", "continuity"])
def test_coupled_transport_scalar(tmpdir, geometry, equation_form1, equation_form2, tracer_setup):
    setup = tracer_setup(tmpdir, geometry)
    domain = setup.domain

    if equation_form1 == "advective":
        tracer1 = ActiveTracer(name='f1', space='DG',
                               variable_type=TracerVariableType.mixing_ratio,
                               transport_eqn=TransportEquationType.advective)
    else:
        tracer1 = ActiveTracer(name='f1', space='DG',
                               variable_type=TracerVariableType.density,
                               transport_eqn=TransportEquationType.conservative)

    if equation_form2 == "advective":
        tracer2 = ActiveTracer(name='f2', space='DG',
                               variable_type=TracerVariableType.mixing_ratio,
                               transport_eqn=TransportEquationType.advective)
    else:
        tracer2 = ActiveTracer(name='f2', space='DG',
                               variable_type=TracerVariableType.density,
                               transport_eqn=TransportEquationType.conservative)

    tracers = [tracer1, tracer2]

    V = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)

    transport_scheme = SSPRK3(domain)
    transport_method = [DGUpwind(eqn, 'f1'), DGUpwind(eqn, 'f2')]

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, transport_scheme, setup.io, time_varying_velocity, transport_method
    )

    # Initial conditions
    timestepper.fields("f1").interpolate(setup.f_init)
    timestepper.fields("f2").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)

    error1, error2 = run(timestepper, setup.tmax, setup.f_end)
    assert error1 < setup.tol, \
        'The transport error for tracer 1 is greater than the permitted tolerance'
    assert error2 < setup.tol, \
        'The transport error for tracer 2 is greater than the permitted tolerance'


@pytest.mark.parametrize("m_X_space", ['DG', 'theta'])
def test_conservative_coupled_transport(tmpdir, m_X_space, tracer_setup):

    setup = tracer_setup(tmpdir, "slice")
    domain = setup.domain
    mesh = domain.mesh

    rho_d = ActiveTracer(name='f1', space='DG',
                         variable_type=TracerVariableType.density,
                         transport_eqn=TransportEquationType.conservative)

    m_X = ActiveTracer(name='f2', space=m_X_space,
                       variable_type=TracerVariableType.mixing_ratio,
                       transport_eqn=TransportEquationType.tracer_conservative,
                       density_name='f1')

    tracers = [rho_d, m_X]

    V = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)

    if m_X_space == 'theta':
        V_m_X = domain.spaces(m_X_space)
        Vt_brok = FunctionSpace(mesh, BrokenElement(V_m_X.ufl_element()))
        suboptions = {'f1': RecoveryOptions(embedding_space=Vt_brok,
                                            recovered_space=V_m_X,
                                            project_low_method='recover'),
                      'f2': EmbeddedDGOptions()}
        opts = MixedFSOptions(suboptions=suboptions)

        transport_scheme = SSPRK3(
            domain, options=opts,
            rk_formulation=RungeKuttaFormulation.predictor
        )
    else:
        transport_scheme = SSPRK3(
            domain, rk_formulation=RungeKuttaFormulation.predictor
        )

    transport_method = [DGUpwind(eqn, 'f1'), DGUpwind(eqn, 'f2')]

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, transport_scheme, setup.io, time_varying_velocity, transport_method
    )

    # Initial conditions
    timestepper.fields("f1").interpolate(setup.f_init)
    timestepper.fields("f2").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)

    error1, error2 = run(timestepper, setup.tmax, setup.f_end)
    assert error1 < setup.tol, \
        'The transport error for rho_d is greater than the permitted tolerance'
    assert error2 < setup.tol, \
        'The transport error for m_X is greater than the permitted tolerance'
