"""
Test the formulae for finding shortest distances between two points.
"""
import importlib
import numpy as np
import firedrake as fd
from gusto.coord_transforms import *
import pytest

tol = 1e-12

# Structure of values for testing Firedrake and numpy routines are different
def setup_values(values_list, module_name, mesh=None):

    if module_name == "firedrake":
        # Put data into fields
        DG0 = fd.FunctionSpace(mesh, 'DG', 0)
        new_values = fd.Function(DG0)
        new_values.dat.data[:] = values_list[:]
    else:
        new_values = np.array(values_list)
    return new_values


# Checks values and prints an error message if they aren't correct
def check_values(new_values, answers, module_name, routine):

    if module_name == 'firedrake':
        # Unpack Firedrake field to array of data
        V = answers.function_space()
        field = fd.Function(V)
        field.interpolate(new_values)
        new_values = field.dat.data[:]
        answers = answers.dat.data[:]

    assert np.allclose(new_values, answers, atol=tol), \
        f'Incorrect answer for {module_name} module and {routine} '


@pytest.mark.parametrize("module_name", ["numpy", "firedrake"])
def test_great_arc_angle(module_name):

    module = importlib.import_module(module_name)
    pi = module.pi
    sqrt = module.sqrt
    acos = module.acos if hasattr(module, "acos") else module.arccos

    lon2 = pi / 4.0
    lat2 = pi / 4.0
    raw_lon1 = [0.0, 0.0, pi/4, pi/4, pi/4]
    raw_lat1 = [0.0, pi/4, 0.0, pi/2, -pi/2]
    raw_answers = [pi/3, acos(0.5*(1+1/sqrt(2))), pi/4, pi/4, 3*pi/4]

    # Test for firedrake routines requires a mesh
    if module_name == "firedrake":
        mesh = fd.UnitIntervalMesh(len(raw_answers))
    else:
        mesh = None

    # Put the coordinates in numpy arrays or firedrake functions
    lon1 = setup_values(raw_lon1, module_name, mesh=mesh)
    lat1 = setup_values(raw_lat1, module_name, mesh=mesh)
    answers = setup_values(raw_answers, module_name, mesh=mesh)

    distances = great_arc_angle(lon1, lat1, lon2, lat2)

    check_values(distances, answers, module_name, 'great_arc')


@pytest.mark.parametrize("module_name", ["numpy", "firedrake"])
def test_periodic_distance(module_name):

    max_x = 10.0
    x2 = 6.0
    raw_x1 = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    raw_answers = [4.0, -5.0, -4.0, -2.0, -1.0, 0.0, 2.0, 4.0]

    # Test for firedrake routines requires a mesh
    if module_name == "firedrake":
        mesh = fd.UnitIntervalMesh(len(raw_answers))
    else:
        mesh = None

    # Put the coordinates in numpy arrays or firedrake functions
    x1 = setup_values(raw_x1, module_name, mesh=mesh)
    answers = setup_values(raw_answers, module_name, mesh=mesh)

    distances = periodic_distance(x1, x2, max_x)

    check_values(distances, answers, module_name, "periodic_distance")
