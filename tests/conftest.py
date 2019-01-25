from firedrake import (IcosahedralSphereMesh, PeriodicIntervalMesh,
                       ExtrudedMesh, SpatialCoordinate, as_vector,
                       sin, exp)
from gusto import *
from collections import namedtuple
from math import pi
import pytest

AdvectionSetup = namedtuple('AdvectionSetup',
                            ['state', 'dt', 'tmax', 'f_init', 'f_end', 'err'])


def advection_test_sphere(tmpdir):

    mesh = IcosahedralSphereMesh(radius=1,
                                 refinement_level=3,
                                 degree=1)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    output = OutputParameters(dirname=str(tmpdir), dumpfreq=15)
    state = State(mesh, output=output)
    build_spaces(state, "BDM", 1)

    u = state.fields("u", space=state.spaces("HDiv"))
    u.project(as_vector([-x[1], x[0], 0.0]))

    dt = pi/3. * 0.02
    tmax = pi/2
    f_init = exp(-x[2]**2 - x[0]**2)
    f_end = exp(-x[2]**2 - x[1]**2)
    err = 2.5e-2

    return AdvectionSetup(state, dt, tmax, f_init, f_end, err)


def advection_test_slice(tmpdir):
    m = PeriodicIntervalMesh(15, 1.)
    mesh = ExtrudedMesh(m, layers=15, layer_height=1./15.)

    output = OutputParameters(dirname=str(tmpdir), dumpfreq=15)
    state = State(mesh, output=output)
    build_spaces(state, "CG", 1, 1)

    u = state.fields("u", space=state.spaces("HDiv"))
    u.project(as_vector([1.0, 0.0]))

    dt = 0.02
    tmax = 2.5
    x = SpatialCoordinate(mesh)
    f_init = sin(2*pi*x[0])*sin(2*pi*x[1])
    f_end = sin(2*pi*(x[0]-0.5))*sin(2*pi*x[1])
    err = 7e-2

    return AdvectionSetup(state, dt, tmax, f_init, f_end, err)


@pytest.fixture()
def advection_setup(tmpdir):

    def _advection_setup(geometry):
        if geometry == "sphere":
            return advection_test_sphere(tmpdir)
        elif geometry == "slice":
            return advection_test_slice(tmpdir)

    return _advection_setup
