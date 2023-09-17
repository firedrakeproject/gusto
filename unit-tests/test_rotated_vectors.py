"""
Test the formulae for rotating spherical vectors.
"""
import numpy as np
from gusto.coord_transforms import *

tol = 1e-12


def test_rotated_lonlatr_vectors_firedrake():

    from firedrake import (CubedSphereMesh, pi, SpatialCoordinate, Function,
                           VectorFunctionSpace, as_vector, grad, sqrt, dot,
                           atan)

    new_pole = (pi/4, pi/4)

    radius = 10.0
    mesh = CubedSphereMesh(radius)

    xyz = SpatialCoordinate(mesh)

    rot_axis, rot_angle = pole_rotation(new_pole)
    new_xyz = rodrigues_rotation(xyz, as_vector(rot_axis), rot_angle)

    # TODO: this should be
    # new_lonlatr = lonlatr_from_xyz(new_xyz[0], new_xyz[1], new_xyz[2])
    # but when atan_2 became atan2 in UFL we lost the derivative of atan2
    # therefore define this here with atan
    lon = atan(new_xyz[1]/new_xyz[0])
    l = sqrt(new_xyz[0]**2 + new_xyz[1]**2)
    lat = atan(new_xyz[2]/l)


    # Do an alternative calculation based on gradients of new coordinates
    answer_e_lon = grad(lon)
    answer_e_lat = grad(lat)
    answer_e_r = grad(sqrt(dot(xyz, xyz)))

    # Normalise
    answer_e_lon /= sqrt(dot(answer_e_lon, answer_e_lon))
    answer_e_lat /= sqrt(dot(answer_e_lat, answer_e_lat))
    answer_e_r /= sqrt(dot(answer_e_r, answer_e_r))

    new_e_lon, new_e_lat, new_e_r = rotated_lonlatr_vectors(xyz, new_pole)

    # Check answers
    V = VectorFunctionSpace(mesh, "CG", 1)

    for new_vector, answer_vector, component in zip([new_e_lon, new_e_lat, new_e_r],
                                                    [answer_e_lon, answer_e_lat, answer_e_r],
                                                    ['lon', 'lat', 'r']):

        new_field = Function(V).interpolate(new_vector)
        answer_field = Function(V).interpolate(as_vector(answer_vector))

        assert np.allclose(new_field.dat.data, answer_field.dat.data), \
            f'Incorrect answer for firedrake rotated {component} vector'
