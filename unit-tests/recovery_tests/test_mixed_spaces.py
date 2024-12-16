"""
Test that the Recovery_options object builds spaces of the correct degree
"""
from firedrake import UnitIntervalMesh, ExtrudedMesh
from gusto.core import Domain
from gusto.recovery.recovery_options import RecoverySpaces
import pytest

order_correct_degree_dict = {(0, 0): {'DG': (1, 1),
                                      'HDiv': (2, 2),
                                      'theta': (1, 1)},
                             (0, 1): {'DG': (1, 1),
                                      'HDiv': (2, 2),
                                      'theta': (1, 2)},
                             (1, 0): {'DG': (1, 1),
                                      'HDiv': (2, 2),
                                      'theta': (1, 1)}
                             }


@pytest.mark.parametrize("order", [(0, 0), (1, 0), (0, 1)])
@pytest.mark.parametrize('space', ['DG', 'HDiv', 'theta'])
def test_mixed_spaces(order, space):
    mesh = UnitIntervalMesh(3)
    emesh = ExtrudedMesh(mesh, 3, 0.33)
    dt = 1
    domain = Domain(emesh, dt, family='CG', horizontal_degree=order[0], vertical_degree=order[1])
    recovery_spaces = RecoverySpaces(domain)

    tesing_space = getattr(recovery_spaces, f'{space}_options')
    degree = tesing_space.recovered_space.finat_element.degree
    assert degree == order_correct_degree_dict[order][space]
