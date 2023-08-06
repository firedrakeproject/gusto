"""
Test the formulae for transforming between spherical and Cartesian coordinates.
"""
import importlib
import numpy as np
import firedrake as fd
from gusto.coord_transforms import *
import pytest

tol = 1e-12

# Structure of coordinates for testing Firedrake and numpy routines are different
def setup_coordinates(coords_list, module_name, mesh=None):
    # Coords should be a list of lists
    all_coords = np.array(coords_list)
    _, num_coords = np.shape(all_coords)

    if module_name == "firedrake":
        # Put data into fields
        DG0 = fd.FunctionSpace(mesh, 'DG', 0)
        coord_fields = []
        for i in range(num_coords):
            coord_field = fd.Function(DG0)
            coord_field.dat.data[:] = all_coords[:, i]
            coord_fields.append(coord_field)
        return coord_fields
    else:
        # Transform to list of arrays
        new_coords_list = []
        for i in range(num_coords):
            new_coords_list.append(all_coords[:, i])
        return new_coords_list


# Checks coordinate values and prints an error message if they aren't correct
def check_coords(new_values, answers, module_name, routine):

    if module_name == 'firedrake':
        V = answers[0].function_space()
        fields = [fd.Function(V) for _ in new_values]
        for field, new_value in zip(fields, new_values):
            field.interpolate(new_value)
        new_values = [field.dat.data[:] for field in fields]
        answers = [answer.dat.data[:] for answer in answers]

    for i, (new_value, answer) in enumerate(zip(new_values, answers)):
        assert np.all(np.isclose(new_value, answer, atol=tol)), \
            f'Incorrect answer for {module_name} module and {routine} ' \
            + f'routine, coord {i}'


@pytest.mark.parametrize("module_name", ["numpy", "firedrake"])
def test_xyz_and_lonlatr(module_name):

    module = importlib.import_module(module_name)
    pi = module.pi

    # Use the following sets of coordinates:
    # (r, lon, lat)  <--> (x, y, z)
    # (2,0,pi/2)     <--> (0, 0, 2)
    # (0.5,pi,0)     <--> (-0.5, 0, 0)
    # (10,-pi/2,0)   <--> (0,-10, 0)
    # (0,0,0)        <--> (0, 0, 0)

    raw_llr_coords = [[0.0, pi/2, 2.0],
                      [pi, 0.0, 0.5],
                      [-pi/2, 0.0, 10],
                      [0.0, 0.0, 0.0]]

    raw_xyz_coords = [[0.0, 0.0, 2.0],
                      [-0.5, 0.0, 0.0],
                      [0.0, -10.0, 0.0],
                      [0.0, 0.0, 0.0]]

    # Test for firedrake routines requires a mesh
    if module_name == "firedrake":
        mesh = fd.UnitIntervalMesh(len(raw_llr_coords))
    else:
        mesh = None

    # Put the coordinates in numpy arrays or firedrake functions
    llr_coords = setup_coordinates(raw_llr_coords, module_name, mesh=mesh)
    xyz_coords = setup_coordinates(raw_xyz_coords, module_name, mesh=mesh)

    new_llr = lonlatr_from_xyz(xyz_coords[0], xyz_coords[1], xyz_coords[2])
    new_xyz = xyz_from_lonlatr(llr_coords[0], llr_coords[1], llr_coords[2])

    check_coords(new_llr, llr_coords, module_name, 'lonlatr_from_xyz')
    check_coords(new_xyz, xyz_coords, module_name, 'xyz_from_lonlatr')
