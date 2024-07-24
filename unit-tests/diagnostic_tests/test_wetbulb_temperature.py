
from gusto.diagnostics import WetBulbTemperature
from gusto.core.fields import StateFields, PrescribedFields, TimeLevelFields
from gusto import (Domain, CompressibleParameters, CompressibleEulerEquations,
                   WaterVapour)
from firedrake import PeriodicIntervalMesh, ExtrudedMesh
import numpy as np


def test_wetbulb_temperature():

    L = 10
    H = 10
    ncol = 3
    nlayers = 3

    m = PeriodicIntervalMesh(ncol, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    domain = Domain(mesh, 0.1, 'CG', 1)
    params = CompressibleParameters()
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

    diagnostic = WetBulbTemperature(eqn, gamma=0.5, num_iterations=10)
    diagnostic.setup(domain, state_fields)
    diagnostic.compute()

    # Answer computed from online calculator, using the formula:
    # Tw = T × arctan[0.151977 × (RH% + 8.313659)^(1/2)]
    # + arctan(T + RH%) - arctan(RH% - 1.676331)
    # + 0.00391838 ×(RH%)^(3/2) × arctan(0.023101 × RH%) - 4.686035.
    # where T is in degrees C and RH is a percentage

    assert np.allclose(diagnostic.field.dat.data, 286.8, atol=0.5), \
        'The wet-bulb diagnostic does not seem to be correct'
