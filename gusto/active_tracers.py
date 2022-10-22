"""
Defines the :class:`ActiveTracer` object, which contains the metadata to
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
    Denotes the type of the variable describing the tracer.

    An enumerator object which stores the variable type of a tracer X. If the
    density of tracer X is rho_X, the density of dry air is rho_d and the total
    density is rho_t then these variables are given by:

    mixing ratio = rho_X / rho_d
    specific_humidity = rho_X / rho_t
    density = rho_X
    """

    mixing_ratio = 25
    specific_humidity = 644
    density = 137


class Phases(Enum):
    """An enumerator object which describes the phase of a substance."""

    gas = 38
    liquid = 112
    solid = 83
    plasma = 2000  # Why not!


class ActiveTracer(object):
    """
    Object containing metadata to describe an active tracer variable.

    A class containing the metadata to describe how an active tracer variable
    is used within an equation set, being added as a component within the
    :class:`MixedFunctionSpace` as these variables interact with the other
    prognostic variables.
    """
    def __init__(self, name, space, variable_type, transport_flag=True,
                 transport_eqn=TransportEquationType.advective,
                 phase=Phases.gas, is_moisture=False):
        """
        Args:
            name (str): the name for the variable.
            space (str): the name of the :class:`FunctionSpace` for the tracer.
            variable_type (:class:`TracerVariableType`): enumerator indicating
                the type of tracer variable (e.g. mixing ratio or density).
            transport_flag (bool, optional): whether this tracer is to be
                transported or not. Defaults to True.
            transport_eqn (:class:`TransportEquationType`, optional): enumerator
                indicating the type of transport equation to be solved (e.g.
                advective). Defaults to `TransportEquationType.advective`.
            phase (:class:`Phases`, optional): enumerator indicating the phase
                of the tracer variable. Defaults to `Phases.gas`.
            is_moisture (bool, optional): whether the tracer is a water variable
                or not. Defaults to False.

        Raises:
            ValueError: if the `transport_eqn` is `no_transport` but
                `transport_flag` is True.
            ValueError: if `transport_flag` is False but `transport_eqn` is not
                `no_transport`.
            NotImplementedError: if `variable_type` is not `mixing_ratio`.
        """

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
    """An object encoding the details of water vapour as a tracer."""
    def __init__(self, name='vapour', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_flag=True,
                 transport_eqn=TransportEquationType.advective):
        """
        Args:
            name (str, optional): the variable's name. Defaults to 'vapour'.
            space (str, optional): the name for the :class:`FunctionSpace` to be
                used by the variable. Defaults to 'theta'.
            variable_type (:class:`TracerVariableType`, optional): enumerator
                indicating the type of tracer variable (e.g. mixing ratio or
                density). Defaults to `TracerVariableType.mixing_ratio`.
            transport_flag (bool, optional): whether this tracer is to be
                transported or not. Defaults to True.
            transport_eqn (:class:`TransportEquationType`, optional): enumerator
                indicating the type of transport equation to be solved (e.g.
                advective). Defaults to `TransportEquationType.advective`.
        """
        super().__init__(f'{name}_{variable_type.name}', space, variable_type,
                         transport_flag, phase=Phases.gas, is_moisture=True)


class CloudWater(ActiveTracer):
    """An object encoding the details of cloud water as a tracer."""
    def __init__(self, name='cloud_liquid', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_flag=True,
                 transport_eqn=TransportEquationType.advective):
        """
        Args:
            name (str, optional): the variable name. Default is 'cloud_liquid'.
            space (str, optional): the name for the :class:`FunctionSpace` to be
                used by the variable. Defaults to 'theta'.
            variable_type (:class:`TracerVariableType`, optional): enumerator
                indicating the type of tracer variable (e.g. mixing ratio or
                density). Defaults to `TracerVariableType.mixing_ratio`.
            transport_flag (bool, optional): whether this tracer is to be
                transported or not. Defaults to True.
            transport_eqn (:class:`TransportEquationType`, optional): enumerator
                indicating the type of transport equation to be solved (e.g.
                advective). Defaults to `TransportEquationType.advective`.
        """
        super().__init__(f'{name}_{variable_type.name}', space, variable_type,
                         transport_flag, phase=Phases.liquid, is_moisture=True)


class Rain(ActiveTracer):
    """An object encoding the details of rain as a tracer."""
    def __init__(self, name='rain', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_flag=True,
                 transport_eqn=TransportEquationType.advective):
        """
        Args:
            name (str, optional): the name for the variable. Defaults to 'rain'.
            space (str, optional): the name for the :class:`FunctionSpace` to be
                used by the variable. Defaults to 'theta'.
            variable_type (:class:`TracerVariableType`, optional): enumerator
                indicating the type of tracer variable (e.g. mixing ratio or
                density). Defaults to `TracerVariableType.mixing_ratio`.
            transport_flag (bool, optional): whether this tracer is to be
                transported or not. Defaults to True.
            transport_eqn (:class:`TransportEquationType`, optional): enumerator
                indicating the type of transport equation to be solved (e.g.
                advective). Defaults to `TransportEquationType.advective`.
        """
        super().__init__(f'{name}_{variable_type.name}', space, variable_type,
                         transport_flag, phase=Phases.liquid, is_moisture=True)
