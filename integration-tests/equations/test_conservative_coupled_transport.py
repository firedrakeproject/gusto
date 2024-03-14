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
    norm1 = norm(timestepper.fields("rho_d") - f_end) / norm(f_end)
    norm2 = norm(timestepper.fields("m_X") - f_end) / norm(f_end)
    return norm1, norm2

@pytest.mark.parametrize("m_X_space", ['DG', 'theta'])
def test_coupled_transport_scalar(tmpdir, m_X_space, tracer_setup):

    setup = tracer_setup(tmpdir, "slice")
    domain = setup.domain
    mesh = domain.mesh

    rho_d = ActiveTracer(name='rho_d', space='DG',
                         variable_type=TracerVariableType.density,
                         transport_eqn=TransportEquationType.conservative)

    m_X = ActiveTracer(name='m_X', space=m_X_space,
                     variable_type=TracerVariableType.mixing_ratio,
                     transport_eqn=TransportEquationType.tracer_conservative,
                     density_name='rho_d')

    tracers = [rho_d, m_X]

    V = domain.spaces("HDiv")
    eqn = ConservativeCoupledTransportEquation(domain, active_tracers=tracers, Vu=V)

    if m_X_space == 'theta':
        V_m_X = domain.spaces(m_X_space)
        Vt_brok = FunctionSpace(mesh, BrokenElement(V_m_X.ufl_element()))
        suboptions = {'rho_d':RecoveryOptions(embedding_space=Vt_brok,
                                          recovered_space=V_m_X,
                                          project_low_method='recover'),
                      'm_X': EmbeddedDGOptions()}
        opts = MixedFSOptions(suboptions=suboptions)

        transport_scheme = SSPRK3(domain, options = opts, increment_form=False)

    transport_scheme = SSPRK3(domain, increment_form=False)

    transport_method = [DGUpwind(eqn, 'rho_d'), DGUpwind(eqn, 'm_X')]

    timestepper = PrescribedTransport(eqn, transport_scheme, setup.io, transport_method)

    # Initial conditions
    timestepper.fields("rho_d").interpolate(setup.f_init)
    timestepper.fields("m_X").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)

    error1, error2 = run(timestepper, setup.tmax, setup.f_end)
    assert error1 < setup.tol, \
        'The transport error for rho_d is greater than the permitted tolerance'
    assert error2 < setup.tol, \
        'The transport error for m_X is greater than the permitted tolerance'
