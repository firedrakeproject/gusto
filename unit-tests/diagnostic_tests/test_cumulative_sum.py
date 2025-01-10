
from gusto.diagnostics import CumulativeSum
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
    params = CompressibleParameters()
    active_tracers = [WaterVapour()]
    eqn = CompressibleEulerEquations(domain, params, active_tracers=active_tracers)
    prog_fields = TimeLevelFields(eqn)

    Vtheta = domain.spaces('theta')

    # Setting up prognostic fields for the diagnostic to use
    prescribed_fields = PrescribedFields()
    state_fields = StateFields(prog_fields, prescribed_fields)

    theta = state_fields('theta', Vtheta)

    # Initial conditions
    theta.interpolate(300.0)

    diagnostic = CumulativeSum(name='theta')
    diagnostic.setup(domain, state_fields)
    diagnostic.compute()

    assert np.allclose(diagnostic.field.dat.data, 300.0, atol=0.0), \
        'The cumulative sum diagnostic does not seem to be correct after 1 iteration'

    theta.interpolate(250.0)
    diagnostic.compute()

    assert np.allclose(diagnostic.field.dat.data, 550.0, atol=0.0), \
        'The cumulative sum diagnostic does not seem to be correct after 2 iterations'
