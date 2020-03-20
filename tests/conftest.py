from firedrake import (IcosahedralSphereMesh, PeriodicIntervalMesh,
                       ExtrudedMesh, SpatialCoordinate, as_vector,
                       sin, exp, pi)
from gusto import *
from collections import namedtuple
import pytest

opts = ('state', 'tmax', 'f_init', 'f_end', 'tol')
TracerSetup = namedtuple('TracerSetup', opts)
TracerSetup.__new__.__defaults__ = (None,)*len(opts)


def tracer_sphere(tmpdir):
    mesh = IcosahedralSphereMesh(radius=1,
                                 refinement_level=3,
                                 degree=1)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    dt = pi/3. * 0.02
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=15)
    state = State(mesh, horizontal_degree=1, family="BDM", dt=dt,
                  output=output)

    u = state.fields("u", space=state.spaces("HDiv"))
    u.project(as_vector([-x[1], x[0], 0.0]))

    tmax = pi/2
    f_init = exp(-x[2]**2 - x[0]**2)
    f_end = exp(-x[2]**2 - x[1]**2)
    tol = 2.5e-2

    return TracerSetup(state, tmax, f_init, f_end, tol)


def tracer_slice(tmpdir):
    m = PeriodicIntervalMesh(15, 1.)
    mesh = ExtrudedMesh(m, layers=15, layer_height=1./15.)

    dt = 0.02
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=15)
    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG", dt=dt, output=output)

    u = state.fields("u", space=state.spaces("HDiv"))
    u.project(as_vector([1.0, 0.0]))

    tmax = 2.5
    x = SpatialCoordinate(mesh)
    f_init = sin(2*pi*x[0])*sin(2*pi*x[1])
    f_end = sin(2*pi*(x[0]-0.5))*sin(2*pi*x[1])
    tol = 7e-2

    return TracerSetup(state, tmax, f_init, f_end, tol)


def tracer_blob_slice(tmpdir):
    dt = 0.01
    L = 10.
    m = PeriodicIntervalMesh(10, L)
    mesh = ExtrudedMesh(m, layers=10, layer_height=1.)

    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG", dt=dt, output=output)

    tmax = 1.
    x = SpatialCoordinate(mesh)
    f_init = exp(-((x[0]-0.5*L)**2 + (x[1]-0.5*L)**2))

    return TracerSetup(state, tmax, f_init)


@pytest.fixture()
def tracer_setup():

    def _tracer_setup(tmpdir, geometry, blob=False):
        if geometry == "sphere":
            assert not blob
            return tracer_sphere(tmpdir)
        elif geometry == "slice":
            if blob:
                return tracer_blob_slice(tmpdir)
            else:
                return tracer_slice(tmpdir)

    return _tracer_setup
