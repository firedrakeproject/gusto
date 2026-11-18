
from gusto.diagnostics import ShallowWaterAvailablePotentialEnergy
from gusto.core.fields import StateFields, PrescribedFields, TimeLevelFields
from gusto import (Domain, ShallowWaterParameters, ShallowWaterEquations)
from firedrake import PeriodicSquareMesh
import numpy as np


def test_swavlbpe():

    nx = 10
    Lx = 1
    H = 0

    mesh = PeriodicSquareMesh(nx=nx, ny=nx, L=Lx, quadrilateral=True)

    domain = Domain(mesh, 0.1, 'RTCF', 1)
    params = ShallowWaterParameters(mesh, H=H)
    eqn = ShallowWaterEquations(domain, params)
    prog_fields = TimeLevelFields(eqn)
    prescribed_fields = PrescribedFields()
    state_fields = StateFields(prog_fields, prescribed_fields)

    diagnostic = ShallowWaterAvailablePotentialEnergy(params)
    diagnostic.setup(domain, state_fields)
    diagnostic.compute()

    print(diagnostic.field.dat.data)

    g = params.g

    assert np.allclose(diagnostic.field.dat.data, 0.5*g*H**2, atol=0.0), \
        'The sw available potential energy diagnostic does not seem to be correct'
