"""
Tests various Gusto ActiveTracer objects.
"""

from gusto import (ActiveTracer, TransportEquationType,
                   TracerVariableType, Phases)


def test_tracer_classes():

    names = ['mr_v', 'big_blob']
    spaces = ['V', 'U']
    transport_eqns = [TransportEquationType.advective,
                      TransportEquationType.no_transport]
    variable_types = [TracerVariableType.mixing_ratio]

    for name, space, transport_eqn in zip(names, spaces, transport_eqns):

        # Test active tracer class
        for variable_type in variable_types:
            tracer = ActiveTracer(name, space, variable_type, transport_eqn)
            assert tracer.name == name
            assert tracer.space == space
            assert tracer.transport_eqn == transport_eqn
            assert tracer.variable_type == variable_type
            assert tracer.phase == Phases.gas
