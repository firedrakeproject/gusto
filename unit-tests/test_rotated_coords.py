"""
Test the formulae for rotating spherical coordinates.
"""
import numpy as np
from gusto.coord_transforms import *

tol = 1e-12


def test_rotated_lonlatr_coords_numpy():

    pi = np.pi

    rotated_pole = (pi/2, 0.0)

    llr_coords = np.array(([pi, pi/2, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.5, 10.0, 0.0]))

    xyz_coords = np.array(([-0.5, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0]))

    new_llr = rotated_lonlatr_coords(xyz_coords, rotated_pole)

    assert np.allclose(new_llr, llr_coords), \
        'Incorrect answer for numpy and rotated_lonlatr'


def test_rotated_lonlatr_coords_firedrake():

    from firedrake import (CubedSphereMesh, pi, SpatialCoordinate, Function,
                           FunctionSpace)

    rotated_pole = (pi/2, 0.0)

    radius = 10.0
    mesh = CubedSphereMesh(radius)

    xyz = SpatialCoordinate(mesh)
    rotated_llr = rotated_lonlatr_coords(xyz, rotated_pole)

    # Check against new (X,Y,Z)
    new_xyz = xyz_from_lonlatr(rotated_llr[0], rotated_llr[1], rotated_llr[2])

    V = FunctionSpace(mesh, "CG", 1)

    x = Function(V).interpolate(xyz[0])
    y = Function(V).interpolate(xyz[1])
    z = Function(V).interpolate(xyz[2])
    new_x = Function(V).interpolate(new_xyz[0])
    new_y = Function(V).interpolate(new_xyz[1])
    new_z = Function(V).interpolate(new_xyz[2])

    assert np.allclose(x.dat.data, new_x.dat.data), \
        'Incorrect answer for firedrake rotated x coordinates'
    assert np.allclose(y.dat.data, -new_z.dat.data), \
        'Incorrect answer for firedrake rotated y coordinates'
    assert np.allclose(z.dat.data, new_y.dat.data), \
        'Incorrect answer for firedrake rotated z coordinates'
