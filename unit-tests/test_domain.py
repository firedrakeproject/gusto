"""
Tests the identification of different domains in Gusto.
"""

from firedrake import (UnitIntervalMesh, ExtrudedMesh, UnitSquareMesh,
                       UnitIcosahedralSphereMesh)
from gusto import Domain
import pytest


@pytest.mark.parametrize("domain_name", ['interval', 'vertical_slice',
                                         'plane', 'extruded_plane',
                                         'spherical_shell',
                                         'extruded_spherical_shell'])
def test_domain(domain_name):

    dt = 1.0

    if domain_name == 'interval':
        mesh = UnitIntervalMesh(3)
        family = 'CG'
    elif domain_name == 'vertical_slice':
        m = UnitIntervalMesh(3)
        mesh = ExtrudedMesh(m, 3, 3)
        family = 'CG'
    elif domain_name == 'plane':
        mesh = UnitSquareMesh(3, 3)
        family = 'RT'
    elif domain_name == 'extruded_plane':
        m = UnitSquareMesh(3, 3)
        mesh = ExtrudedMesh(m, 3, 3)
        family = 'RT'
    elif domain_name == 'spherical_shell':
        mesh = UnitIcosahedralSphereMesh()
        family = 'RT'
    elif domain_name == 'extruded_spherical_shell':
        m = UnitIcosahedralSphereMesh()
        mesh = ExtrudedMesh(m, 3, 3, extrusion_type='radial')
        family = 'RT'
    else:
        raise ValueError(f'domain {domain_name} not recognised')

    domain = Domain(mesh, dt, family, 1)

    # Check that the Domain correctly works out what kind of domain it is
    assert domain_name == domain.metadata['domain_type'], \
        f'The Domain has not correctly recognised domain {domain_name}'
