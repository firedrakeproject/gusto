"""
This file contains the ActiveTracer class, which contains the metadata to
augment equation sets with additional active tracer variables. Some specific
commonly used tracers are also provided.

Enumerators are also defined to encode different aspects of the tracers (e.g.
what type of variable the tracer is, what phase it is, etc).
"""

from enum import Enum
from gusto.configuration import TransportEquationType

__all__ = ["TracerVariableType", "Phases", "ActiveTracer",
           "WaterVapour", "CloudWater", "Rain"]


class TracerVariableType(Enum):
    """
    An Enum object which stores the variable type of a tracer X. If the density
    of tracer X is \rho_X, the density of dry air is \rho_d and the total
    density is \rho_t then these variables are given by:

    mixing ratio = \rho_X / \rho_d
    specific_humidity = \rho_X / \rho_t
    density = \rho_X
    """

    mixing_ratio = 25
    specific_humidity = 644
    density = 137


class Phases(Enum):
    """
    An Enum object which describes the phase of a substance.
    """

    gas = 38
    liquid = 112
    solid = 83
    plasma = 2000  # Why not!


class ActiveTracer(object):
    """
    A class containing the metadata to describe how an active tracer variable
    is used within an equation set, being added as a component within the
    MixedFunctionSpace as these variables interact strongly with the other
    prognostic variables.

    :arg name:           A string naming the tracer field.
    :arg space:          A string indicating the function space for the variable.
    :arg variable_type:  A TracerVariableType Enum indicating the type of tracer
                         variable (e.g. mixing ratio or density).
    :arg transport_flag: A Boolean indicating if the variable is transported.
    :arg transport_eqn:  A TransportEquationType Enum indicating the form of
                         the transport equation to be used.
    :arg phase:          A Phases Enum indicating the phase of the variable.
    :arg is_moisture:    A Boolean indicating whether the variable is moisture.
    """
    def __init__(self, name, space, variable_type, transport_flag=True,
                 transport_eqn=TransportEquationType.advective,
                 phase=Phases.gas, is_moisture=False):

        if transport_flag and transport_eqn == TransportEquationType.no_transport:
            raise ValueError('If tracer is to be transported, transport_eqn must be specified')
        elif not transport_flag and transport_eqn != TransportEquationType.no_transport:
            raise ValueError('If tracer is not to be transported, transport_eqn must be no_transport')

        self.name = name
        self.space = space
        self.transport_flag = transport_flag
        self.transport_eqn = transport_eqn
        self.variable_type = variable_type
        self.phase = phase
        self.is_moisture = is_moisture
        if self.variable_type != TracerVariableType.mixing_ratio:
            raise NotImplementedError('Only mixing ratio tracers are currently implemented')


class WaterVapour(ActiveTracer):
    """
    An object encoding the details of water vapour as a tracer.
    """
    def __init__(self, name='vapour', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_flag=True,
                 transport_eqn=TransportEquationType.advective):
        super().__init__(f'{name}_{variable_type.name}', space, variable_type,
                         transport_flag, phase=Phases.gas, is_moisture=True)


class CloudWater(ActiveTracer):
    """
    An object encoding the details of cloud water as a tracer.
    """
    def __init__(self, name='cloud_liquid', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_flag=True,
                 transport_eqn=TransportEquationType.advective):
        super().__init__(f'{name}_{variable_type.name}', space, variable_type,
                         transport_flag, phase=Phases.liquid, is_moisture=True)


class Rain(ActiveTracer):
    """
    An object encoding the details of rain as a tracer.
    """
    def __init__(self, name='rain', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_flag=True,
                 transport_eqn=TransportEquationType.advective):
        super().__init__(f'{name}_{variable_type.name}', space, variable_type,
                         transport_flag, phase=Phases.liquid, is_moisture=True)
