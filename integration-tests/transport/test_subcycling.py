"""
This tests transport using the subcycling option.

THOUGHTS: this doesn't check that the transport is correct
"""

from gusto import *
from firedrake import PeriodicSquareMesh, exp, SpatialCoordinate, Constant


def setup_gaussian(dirname):
    n = 16
    L = 1.
    mesh = PeriodicSquareMesh(n, n, L)

    parameters = ShallowWaterParameters(H=1.0, g=1.0)
    dt = 0.08
    output = OutputParameters(dirname=dirname+'/sw_plane_gaussian_subcycled')

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters)

    eqns = ShallowWaterEquations(state, family="BDM", degree=1,
                                 fexpr=Constant(1.))
    D0 = state.fields("D")
    x, y = SpatialCoordinate(mesh)
    H = Constant(state.parameters.H)
    D0.interpolate(H + exp(-50*((x-0.5)**2 + (y-0.5)**2)))

    transported_fields = []
    transported_fields.append((SSPRK3(state, "u", options=EmbeddedDGOptions(), subcycles=2)))
    transported_fields.append((SSPRK3(state, "D", subcycles=2)))

    # build time stepper
    stepper = CrankNicolson(state, eqns, transported_fields)

    return stepper


def run(dirname):
    stepper = setup_gaussian(dirname)
    stepper.run(t=0, tmax=0.3)


def test_subcycling(tmpdir):
    dirname = str(tmpdir)
    run(dirname)
