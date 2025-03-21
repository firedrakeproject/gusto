
from gusto.diagnostics import DewpointTemperature
from gusto.core.fields import StateFields, PrescribedFields, TimeLevelFields
from gusto import (Domain, CompressibleParameters, CompressibleEulerEquations,
                   WaterVapour)
from firedrake import PeriodicIntervalMesh, ExtrudedMesh
import numpy as np


def test_dewpoint():

    L = 10
    H = 10
    ncol = 3
    nlayers = 3

    m = PeriodicIntervalMesh(ncol, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    domain = Domain(mesh, 0.1, 'CG', 1)
    params = CompressibleParameters(mesh)
    active_tracers = [WaterVapour()]
    eqn = CompressibleEulerEquations(domain, params, active_tracers=active_tracers)
    prog_fields = TimeLevelFields(eqn)

    DG = domain.spaces('DG')
    Vtheta = domain.spaces('theta')

    # Setting up prognostic fields for the diagnostic to use
    prescribed_fields = PrescribedFields()
    state_fields = StateFields(prog_fields, prescribed_fields)

    theta = state_fields('theta', Vtheta)
    rho = state_fields('rho', DG)
    m_v = state_fields('water_vapour', Vtheta)

    # Initial conditions
    theta.interpolate(300.0)
    rho.interpolate(1.1)
    m_v.interpolate(0.01)

    # This corresponds to:
    # temperature of 288.9 K
    # pressure of 92673 Pa
    # relative humidity of 81.8%

    diagnostic = DewpointTemperature(eqn, num_iterations=20)
    diagnostic.setup(domain, state_fields)
    diagnostic.compute()

    # Answer computed from online calculator, using the formula:
    # T_D = (b × α(T,RH)) / (a - α(T,RH))
    # α(T,RH) = ln(RH/100) + aT/(b+T)
    # where a = 17.625 and b = 243.04 °C
    # T is in degrees C and RH is a percentage

    assert np.allclose(diagnostic.field.dat.data, 285.8, atol=0.5), \
        'The dewpoint diagnostic does not seem to be correct'
