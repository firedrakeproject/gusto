from firedrake import Ensemble, COMM_WORLD, PeriodicUnitSquareMesh
from gusto import *
import pytest


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("spatial_parallelism", [True, False])
def test_parallel_io(tmpdir, spatial_parallelism):

    if spatial_parallelism:
        ensemble = Ensemble(COMM_WORLD, 2)
    else:
        ensemble = Ensemble(COMM_WORLD, 1)

    mesh = PeriodicUnitSquareMesh(10, 10, comm=ensemble.comm)
    dt = 0.1
    domain = Domain(mesh, dt, "BDM", 1)

    # Equation
    parameters = ShallowWaterParameters(H=100)
    equation = ShallowWaterEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=str(tmpdir))
    io = IO(domain, output)

    # Time stepper
    spatial_methods = [DGUpwind(equation, "u"), DGUpwind(equation, "D")]
    stepper = Timestepper(equation, SSPRK3(domain), io, spatial_methods,
                          ensemble=ensemble)

    stepper.run(0, 3*dt)
