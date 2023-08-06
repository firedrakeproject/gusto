"""
Defines the :class:`ActiveTracer` object, which contains the metadata to
augment equation sets with additional active tracer variables. Some specific
commonly used tracers are also provided.

Enumerators are also defined to encode different aspects of the tracers (e.g.
what type of variable the tracer is, what phase it is, etc).
"""

from enum import Enum
from gusto.configuration import TransportEquationType
from gusto.logging import logger

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
    def __init__(self, name, space, variable_type,
                 transport_eqn=TransportEquationType.advective,
                 phase=Phases.gas, chemical=None):
        """
        Args:
            name (str): the name for the variable.
            space (str): the name of the :class:`FunctionSpace` for the tracer.
            variable_type (:class:`TracerVariableType`): enumerator indicating
                the type of tracer variable (e.g. mixing ratio or density).
            transport_eqn (:class:`TransportEquationType`, optional): enumerator
                indicating the type of transport equation to be solved (e.g.
                advective). Defaults to `TransportEquationType.advective`.
            phase (:class:`Phases`, optional): enumerator indicating the phase
                of the tracer variable. Defaults to `Phases.gas`.
            chemical (str, optional): string to describe the chemical that this
                active tracer describes. Defaults to None.

        Raises:
            NotImplementedError: if `variable_type` is not `mixing_ratio`.
        """

        self.name = name
        self.space = space
        self.transport_eqn = transport_eqn
        self.variable_type = variable_type
        self.phase = phase
        self.chemical = chemical
        if self.variable_type != TracerVariableType.mixing_ratio:
            raise NotImplementedError('Only mixing ratio tracers are currently implemented')

        if (variable_type == TracerVariableType.density and transport_eqn == TransportEquationType.advective):
            logger.warning('Active tracer initialised which describes a '
                           + 'density but solving the advective transport eqn')


class WaterVapour(ActiveTracer):
    """An object encoding the details of water vapour as a tracer."""
    def __init__(self, name='water_vapour', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_eqn=TransportEquationType.advective):
        """
        Args:
            name (str, optional): the variable's name. Defaults to
                'water_vapour'.
            space (str, optional): the name for the :class:`FunctionSpace` to be
                used by the variable. Defaults to 'theta'.
            variable_type (:class:`TracerVariableType`, optional): enumerator
                indicating the type of tracer variable (e.g. mixing ratio or
                density). Defaults to `TracerVariableType.mixing_ratio`.
            transport_eqn (:class:`TransportEquationType`, optional): enumerator
                indicating the type of transport equation to be solved (e.g.
                advective). Defaults to `TransportEquationType.advective`.
        """
        super().__init__(f'{name}', space, variable_type,
                         transport_eqn=transport_eqn, phase=Phases.gas, chemical='H2O')


class CloudWater(ActiveTracer):
    """An object encoding the details of cloud water as a tracer."""
    def __init__(self, name='cloud_water', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_eqn=TransportEquationType.advective):
        """
        Args:
            name (str, optional): the variable name. Default is 'cloud_water'.
            space (str, optional): the name for the :class:`FunctionSpace` to be
                used by the variable. Defaults to 'theta'.
            variable_type (:class:`TracerVariableType`, optional): enumerator
                indicating the type of tracer variable (e.g. mixing ratio or
                density). Defaults to `TracerVariableType.mixing_ratio`.
            transport_eqn (:class:`TransportEquationType`, optional): enumerator
                indicating the type of transport equation to be solved (e.g.
                advective). Defaults to `TransportEquationType.advective`.
        """
        super().__init__(f'{name}', space, variable_type,
                         transport_eqn=transport_eqn, phase=Phases.liquid, chemical='H2O')


class Rain(ActiveTracer):
    """An object encoding the details of rain as a tracer."""
    def __init__(self, name='rain', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_eqn=TransportEquationType.advective):
        """
        Args:
            name (str, optional): the name for the variable. Defaults to 'rain'.
            space (str, optional): the name for the :class:`FunctionSpace` to be
                used by the variable. Defaults to 'theta'.
            variable_type (:class:`TracerVariableType`, optional): enumerator
                indicating the type of tracer variable (e.g. mixing ratio or
                density). Defaults to `TracerVariableType.mixing_ratio`.
            transport_eqn (:class:`TransportEquationType`, optional): enumerator
                indicating the type of transport equation to be solved (e.g.
                advective). Defaults to `TransportEquationType.advective`.
        """
        super().__init__(f'{name}', space, variable_type,
                         transport_eqn=transport_eqn, phase=Phases.liquid, chemical='H2O')
