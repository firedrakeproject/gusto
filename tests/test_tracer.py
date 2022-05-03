"""
Tests various Gusto Tracer objects.
"""

from gusto import (Tracer, PassiveTracer, ActiveTracer,
                   TransportEquationForm, TracerVariableType, Phases)
import pytest

def test_tracer_classes():

    names = ['mr_v', 'big_blob']
    spaces = ['V', 'U']
    transport_flags = [True, False]
    transport_eqns = [TransportEquationForm.advective,
                      TransportEquationForm.no_transport]
    variable_types = [TracerVariableType.mixing_ratio]

    for name, space, transport_flag, transport_eqn in \
        zip(names, spaces, transport_flags, transport_eqns):

        # Test Tracer base class
        tracer = Tracer(name, space, transport_flag, transport_eqn)
        assert tracer.name == name
        assert tracer.space == space
        assert tracer.transport_flag == transport_flag
        assert tracer.transport_eqn == transport_eqn

        # Test PassiveTracer base class
        tracer = PassiveTracer(name, space, transport_flag, transport_eqn)
        assert tracer.name == name
        assert tracer.space == space
        assert tracer.transport_flag == transport_flag
        assert tracer.transport_eqn == transport_eqn

        for variable_type in variable_types:
            tracer = ActiveTracer(name, space, variable_type,
                                  transport_flag, transport_eqn)
            assert tracer.name == name
            assert tracer.space == space
            assert tracer.transport_flag == transport_flag
            assert tracer.transport_eqn == transport_eqn
            assert tracer.variable_type == variable_type
            assert tracer.phase == Phases.gas
