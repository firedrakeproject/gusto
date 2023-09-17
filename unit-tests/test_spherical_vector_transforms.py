"""
Test the formulae for transforming vectors between spherical and Cartesian
components. This is tested for numpy arrays, tuples of Firedrake scalars and
Firedrake vector function spaces.
"""
import importlib
import numpy as np
import firedrake as fd
from gusto.coord_transforms import *
import pytest

tol = 1e-12


# Structure of values for testing Firedrake and numpy routines are different
def setup_values(values_list, config_name, mesh=None):
    # Values should be a list of lists
    all_values = np.array(values_list)
    _, num_coords = np.shape(all_values)

    if config_name == "firedrake_scalars":
        # Put data into a series of scalar fields
        DG0 = fd.FunctionSpace(mesh, 'DG', 0)
        scalar_fields = []
        for i in range(num_coords):
            scalar_field = fd.Function(DG0)
            scalar_field.dat.data[:] = all_values[:, i]
            scalar_fields.append(scalar_field)
        return scalar_fields
    elif config_name == "firedrake_vectors":
        # Put data into a single vector-valued field
        Vec_DG0 = fd.VectorFunctionSpace(mesh, 'DG', 0, dim=3)
        vector_field = fd.Function(Vec_DG0)
        vector_field.dat.data[:, :] = all_values[:, :]
        return vector_field

    else:
        # Transform to list of arrays
        new_values_list = []
        for i in range(num_coords):
            new_values_list.append(all_values[:, i])
        return new_values_list


# Checks values and prints an error message if they aren't correct
def check_values(new_values, answers, config_name, routine):

    if config_name == 'firedrake_scalars':
        V = answers[0].function_space()
        fields = [fd.Function(V) for _ in new_values]
        for field, new_value in zip(fields, new_values):
            field.interpolate(new_value)
        new_values = [field.dat.data[:] for field in fields]
        answers = [answer.dat.data[:] for answer in answers]

    elif config_name == 'firedrake_vectors':
        # Interpolate answer into vector function space
        vector_field = fd.Function(answers.function_space())
        vector_field.interpolate(new_values)

        answer_shape = np.shape(answers.dat.data)
        new_values = [vector_field.dat.data[:, i] for i in range(answer_shape[1])]
        answers = [answers.dat.data[:, i] for i in range(answer_shape[1])]

    for i, (new_value, answer) in enumerate(zip(new_values, answers)):
        assert np.all(np.isclose(new_value, answer, atol=tol)), \
            f'Incorrect answer for {config_name} module and {routine} ' \
            + f'routine, coord {i}'


@pytest.mark.parametrize("config_name", ["numpy", "firedrake_scalars", "firedrake_vectors"])
@pytest.mark.parametrize("position_units", ["xyz", "lonlatr_rad"])
def test_xyz_and_lonlatr_vectors(config_name, position_units):

    module_name = "numpy" if config_name == "numpy" else "firedrake"
    coord_config = "numpy" if config_name == "numpy" else "firedrake_scalars"
    module = importlib.import_module(module_name)
    pi = module.pi

    # Consider the following vectors:
    # (r,lon,lat) components  <--> (x,y,z) components at (x,y,z) or (r,lon,lat)
    # (10,-6,0.5)             <--> (10,-6,0.5)        at (5,0,0) or (5,0,0)
    # (0.7,3,1.2)             <--> (3,-0.7,1.2)       at (0,-0.5,0) or (0.5,-pi/2,0)
    # (2,0,5)                 <--> (5,0,-2)           at (0,0,-15) or (15,0,-pi/2)

    raw_llr_coords = [[0.0, 0.0, 5.0],
                      [-pi/2, 0.0, 0.5],
                      [0.0, -pi/2, 15.0],
                      [0.0, 0.0, 0.0]]

    raw_xyz_coords = [[5.0, 0.0, 0.0],
                      [0.0, -0.5, 0.0],
                      [0.0, 0.0, -15.0],
                      [0.0, 0.0, 0.0]]

    raw_llr_vectors = [[-6.0, 0.5, 10.0],
                       [3.0, 1.2, 0.7],
                       [0.0, 5.0, 2.0],
                       [0.0, 0.0, 0.0]]

    raw_xyz_vectors = [[10.0, -6.0, 0.5],
                       [3.0, -0.7, 1.2],
                       [5.0, 0.0, -2.0],
                       [0.0, 0.0, 0.0]]

    # Test for firedrake routines requires a mesh
    if module_name == "firedrake":
        mesh = fd.UnitIntervalMesh(len(raw_llr_coords))
    else:
        mesh = None

    # Put the coordinates in numpy arrays or firedrake functions
    # Coordinates are Firedrake scalars when testing Firedrake vectors
    llr_coords = setup_values(raw_llr_coords, coord_config, mesh=mesh)
    xyz_coords = setup_values(raw_xyz_coords, coord_config, mesh=mesh)
    llr_vectors = setup_values(raw_llr_vectors, config_name, mesh=mesh)
    xyz_vectors = setup_values(raw_xyz_vectors, config_name, mesh=mesh)

    position_vector = xyz_coords if position_units == 'xyz' else llr_coords

    new_llr_vectors = lonlatr_components_from_xyz(xyz_vectors, position_vector, position_units)
    new_xyz_vectors = xyz_vector_from_lonlatr(llr_vectors[0], llr_vectors[1], llr_vectors[2],
                                              position_vector, position_units)

    if config_name != 'numpy':
        llr_vectors = fd.as_vector(llr_vectors)
        new_llr_vectors = fd.as_vector(new_llr_vectors)

    check_values(new_llr_vectors, llr_vectors, config_name, 'lonlatr_components_from_xyz')
    check_values(new_xyz_vectors, xyz_vectors, config_name, 'xyz_vector_from_lonlatr')
