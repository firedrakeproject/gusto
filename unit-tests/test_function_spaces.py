"""
Tests the building of different spaces in the de Rham complex and other useful
spaces in Gusto.
"""

from firedrake import UnitIntervalMesh, ExtrudedMesh, UnitSquareMesh
from gusto import Spaces
import pytest

# List all allowed families for a particular domain
# Don't need to bother testing sphere
domain_family_dict = {'interval': ['CG'],
                      'vertical_slice': ['CG'],
                      'plane': ['BDM', 'BDMF', 'BDME', 'BDFM',
                                'RT', 'RTF', 'RTE', 'RTCF', 'RTCE',
                                'BDMCE', 'BDMCF'],
                      'extruded_plane': ['BDM', 'BDMF', 'BDME', 'BDFM',
                                         'RT', 'RTF', 'RTE', 'RTCF', 'RTCE',
                                         'BDMCE', 'BDMCF']}

reduced_domain_family_dict = {'interval': ['CG'],
                              'vertical_slice': ['CG'],
                              'plane': ['BDM', 'BDFM', 'RT', 'RTCF'],
                              'extruded_plane': ['BDM', 'BDFM', 'RT', 'RTCF']}


# Routine to form all combinations of domains and families
def combos(domains_and_families):
    all_combos = []
    for domain, families in domains_and_families.items():
        for family in families:
            all_combos.append((domain, family))

    return all_combos


def set_up_mesh(domain, family):

    if family in ['BDM', 'BDMF', 'BDME', 'BDFM', 'RT', 'RTE', 'RTF']:
        quadrilateral = False
    elif family in ['RTCF', 'RTCE', 'BDMCE', 'BDMCF']:
        quadrilateral = True

    if domain == 'interval':
        mesh = UnitIntervalMesh(3)
    elif domain == 'vertical_slice':
        m = UnitIntervalMesh(3)
        mesh = ExtrudedMesh(m, 3, 3)
    elif domain == 'plane':
        mesh = UnitSquareMesh(3, 3, quadrilateral=quadrilateral)
    elif domain == 'extruded_plane':
        m = UnitSquareMesh(3, 3, quadrilateral=quadrilateral)
        mesh = ExtrudedMesh(m, 3, 3)
    else:
        raise ValueError(f'domain {domain} not recognised')

    return mesh


# ---------------------------------------------------------------------------- #
# Test creation of full de Rham complex
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("domain, family", combos(domain_family_dict))
def test_de_rham_spaces(domain, family):

    mesh = set_up_mesh(domain, family)
    spaces = Spaces(mesh)

    if domain in ['vertical_slice', 'extruded_plane']:
        # Need horizontal and vertical degrees
        degree = (1, 2)
        spaces.build_compatible_spaces(family, degree[0], degree[1])
    else:
        degree = 1
        spaces.build_compatible_spaces(family, degree)

    # Work out correct CG degree
    if family in ['BDM', 'BDME', 'BDMF', 'BDMCE', 'BDMCF']:
        if domain in ['vertical_slice', 'extruded_plane']:
            cg_degree = (degree[0]+2, degree[1]+1)
        else:
            cg_degree = degree + 2
    elif family == 'BDFM' and domain in ['vertical_slice', 'extruded_plane']:
        cg_degree = (degree[0] + 2, degree[1]+1)
    elif family == 'BDFM':
        cg_degree = degree + 2
    elif domain in ['vertical_slice', 'extruded_plane']:
        cg_degree = (degree[0]+1, degree[1]+1)
    else:
        cg_degree = degree + 1

    # Check that H1 spaces and L2 spaces have the correct degrees
    cg_space = spaces('H1')
    elt = cg_space.ufl_element()
    assert elt.degree() == cg_degree, '"H1" space does not seem to be degree ' \
        + f'{cg_degree}. Found degree {elt.degree()}'

    dg_space = spaces('L2')
    elt = dg_space.ufl_element()
    assert elt.degree() == degree, '"L2" space does not seem to be degree ' \
        + f'{degree}. Found degree {elt.degree()}'

    # Check that continuities have been recorded correctly
    if hasattr(mesh, "_base_mesh"):
        expected_continuity = {
            "H1": {'horizontal': True, 'vertical': True},
            "L2": {'horizontal': False, 'vertical': False},
            "HDiv": {'horizontal': True, 'vertical': True},
            "HCurl": {'horizontal': True, 'vertical': True},
            "theta": {'horizontal': False, 'vertical': True}
        }
    else:
        expected_continuity = {
            "H1": True,
            "L2": False,
            "HDiv": True,
            "HCurl": True
        }

    for space, continuity in expected_continuity.items():
        if space in spaces.continuity:
            assert spaces.continuity[space] == continuity


# ---------------------------------------------------------------------------- #
# Test creation of DG1 equispaced
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("domain, family", combos(reduced_domain_family_dict))
def test_dg_equispaced(domain, family):

    mesh = set_up_mesh(domain, family)
    spaces = Spaces(mesh)
    spaces.build_dg1_equispaced()

    DG1 = spaces('DG1_equispaced')
    elt = DG1.ufl_element()
    assert elt.degree() in [1, (1, 1)], '"DG1 equispaced" does not seem to be ' \
        + f'degree 1. Found degree {elt.degree()}'
    assert elt.variant() == "equispaced", '"DG1 equispaced" does not seem to ' \
        + f'be equispaced variant. Found variant {elt.variant()}'

    if hasattr(mesh, "_base_mesh"):
        expected_continuity = {'horizontal': False, 'vertical': False}
    else:
        expected_continuity = False

    assert spaces.continuity['DG1_equispaced'] == expected_continuity, "DG is discontinuous"


# ---------------------------------------------------------------------------- #
# Test creation of DG0 space
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("domain, family", combos(reduced_domain_family_dict))
def test_dg0(domain, family):

    mesh = set_up_mesh(domain, family)
    spaces = Spaces(mesh)

    DG0 = spaces.create_space('DG0', 'DG', degree=0)
    elt = DG0.ufl_element()
    assert elt.degree() in [0, (0, 0)], '"DG0" space does not seem to be' \
        + f'degree 0. Found degree {elt.degree()}'

    if hasattr(mesh, "_base_mesh"):
        expected_continuity = {'horizontal': False, 'vertical': False}
    else:
        expected_continuity = False

    assert spaces.continuity['DG0'] == expected_continuity, "DG is discontinuous"


# ---------------------------------------------------------------------------- #
# Test creation of a general CG space
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("domain, family", combos(reduced_domain_family_dict))
def test_cg(domain, family):

    mesh = set_up_mesh(domain, family)
    spaces = Spaces(mesh)
    degree = 3

    CG = spaces.create_space('CG', 'CG', degree=degree)

    elt = CG.ufl_element()
    assert elt.degree() == degree or elt.degree() == (degree, degree), \
        (f'"CG" space does not seem to be degree {degree}. '
         + f'Found degree {elt.degree()}')

    if hasattr(mesh, "_base_mesh"):
        expected_continuity = {'horizontal': True, 'vertical': True}
    else:
        expected_continuity = True

    assert spaces.continuity['CG'] == expected_continuity, "CG is continuous"
