"""
Test the formulae for rotating spherical vectors.
"""
import importlib
import numpy as np
import firedrake as fd
from gusto.coord_transforms import *
import pytest

tol = 1e-12

# Structure of values for testing Firedrake and numpy routines are different
def setup_values(values, config_name, len_array, mesh=None):
    if config_name == "numpy_vector":
        # Transform to list of arrays
        vector = np.array(values)
        return vector

    elif config_name == "numpy_field":
        # Transform to list of arrays
        vector_field = np.zeros((len_array, 3))
        for i in range(len_array):
            vector_field[i, :] = values[:]
        return vector_field

    else:
        # Firedrake
        # Put data into a single vector-valued field
        Vec_DG0 = fd.VectorFunctionSpace(mesh, 'DG', 0, dim=3)
        vector_field = fd.Function(Vec_DG0)
        for i in range(len_array):
            vector_field.dat.data[i, :] = values[:]
        return vector_field


def test_pole_rotation():

    pi = np.pi

    new_poles = [(0.0, pi/2), (0.0, 0.0), (pi/2, 0.0)]
    answer_axes = [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)]
    answer_angles = [0.0, pi/2, pi/2]

    for i, (new_pole, answer_axis, answer_angle) in \
            enumerate(zip(new_poles, answer_axes, answer_angles)):
        rot_axis, rot_angle = pole_rotation(new_pole)

        assert abs(rot_angle - answer_angle) < tol, f'pole_rotation {i}: ' \
            + f'rotation angle not correct. Got {rot_angle} but expected {answer_angle}'

        assert np.allclose(np.array(rot_axis), np.array(answer_axis), atol=tol), \
            f'pole_rotation {i}: rotation axis not correct'


# Checks values and prints an error message if they aren't correct
def check_values(new_value, answer, config_name, routine):

    if config_name == 'firedrake':
        # Need to interpolate expression into vector field
        new_field = fd.Function(answer.function_space())
        new_field.interpolate(new_value)
        assert np.allclose(new_field.dat.data, answer.dat.data), \
            f'Incorrect answer for {config_name} and {routine}'
    else:
        assert np.allclose(new_value, answer), \
            f'Incorrect answer for {config_name} and {routine}'


@pytest.mark.parametrize("config_name", ["numpy_field", "numpy_vector", "firedrake"])
def test_rodrigues_rotation(config_name):

    module_name = "firedrake" if config_name == "firedrake" else "numpy"
    module = importlib.import_module(module_name)
    pi = module.pi
    sqrt = module.sqrt
    len_array = 5

    rot_axes = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)]
    rot_angles = [0.0, pi/4, pi/2, pi/2]

    raw_orig_vectors = [(19.4, -14.6, 12.1),
                        (2.0, 0.0, -7.5),
                        (-101.4, 0.6, 7.0),
                        (12.3, -4.5, 0.09)]

    raw_answer_vectors = [(19.4, -14.6, 12.1),
                          (sqrt(2.0), sqrt(2.0), -7.5),
                          (7.0, 0.6, 101.4),
                          (12.3, 0.09, 4.5)]

    # Test for firedrake routines requires a mesh
    if module_name == "firedrake":
        mesh = fd.UnitIntervalMesh(len_array)
    else:
        mesh = None

    for j, (raw_orig_vector, raw_answer_vector, rot_axis, rot_angle) in \
            enumerate(zip(raw_orig_vectors, raw_answer_vectors, rot_axes, rot_angles)):

        if module_name == 'firedrake':
            rot_axis = fd.as_vector(rot_axis)
        else:
            rot_axis = np.array(rot_axis)

        # Put the coordinates in numpy arrays or firedrake functions
        orig_vector = setup_values(raw_orig_vector, config_name, len_array, mesh=mesh)
        answer_vector = setup_values(raw_answer_vector, config_name, len_array, mesh=mesh)

        new_vector = rodrigues_rotation(orig_vector, rot_axis, rot_angle)

        check_values(new_vector, answer_vector, config_name, f'rodrigues_rotation {j}')




