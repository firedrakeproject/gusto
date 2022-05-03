from enum import Enum

__all__ = ["TransportEquationForm", "TracerVariableType", "Phases",
           "Tracer", "PassiveTracer", "ActiveTracer", "WaterVapour",
           "CloudWater", "Rain"]


# TODO: Move this to somewhere else
class TransportEquationForm(Enum):
    """
    An Enum object which stores the forms of the transport equation. For
    transporting velocity 'u' and transported quantity 'q', these equations are:

    advective: dq / dt + dot(u, grad(q)) = 0
    conservative: dq / dt + div(q*u) = 0
    """

    no_transport = 702
    advective = 19
    conservative = 291


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


class Tracer(object):
    """
    Base class for declaring the metadata for a tracer variable.

    :arg name:           A string naming the tracer field.
    :arg space:          A string indicating the function space for the variable.
    :arg transport_flag: A Boolean indicating if the variable is transported.
    :arg transport_eqn:  A TransportEquationForm Enum indicating the form of
                         the transport equation to be used.
    """
    def __init__(self, name, space, transport_flag,
                 transport_eqn=TransportEquationForm.no_transport):

        if transport_flag and transport_eqn == TransportEquationForm.no_transport:
            raise ValueError('If tracer is to be transported, transport_eqn must be specified')
        elif not transport_flag and transport_eqn != TransportEquationForm.no_transport:
            raise ValueError('If tracer is not to be transported, transport_eqn must be no_transport')

        self.name = name
        self.space = space
        self.transport_flag = transport_flag
        self.transport_eqn = transport_eqn


class PassiveTracer(Tracer):
    """
    A class containing the metadata for a passive tracer variable. This
    variable does not feed back onto the prognostic variables.

    :arg name:           A string naming the tracer field.
    :arg space:          A string indicating the function space for the variable.
    :arg transport_flag: A Boolean indicating if the variable is transported.
    :arg transport_eqn:  A TransportEquationForm Enum indicating the form of
                         the transport equation to be used.
    """
    def __init__(self, name, space, transport_flag=True,
                 transport_eqn=TransportEquationForm.advective):
        super().__init__(name, space, transport_flag, transport_eqn)


class ActiveTracer(Tracer):
    """
    A class containing the metadata for an active tracer variable, which
    interacts with the other variables.

    :arg name:           A string naming the tracer field.
    :arg space:          A string indicating the function space for the variable.
    :arg variable_type:  A TracerVariableType Enum indicating the type of tracer
                         variable (e.g. mixing ratio or density).
    :arg transport_flag: A Boolean indicating if the variable is transported.
    :arg transport_eqn:  A TransportEquationForm Enum indicating the form of
                         the transport equation to be used.
    :arg phase:          A Phases Enum indicating the phase of the variable.
    :arg is_moisture:    A Boolean indicating whether the variable is moisture.
    """
    def __init__(self, name, space, variable_type, transport_flag=True,
                 transport_eqn=TransportEquationForm.advective,
                 phase=Phases.gas, is_moisture=False):
        super().__init__(name, space, transport_flag, transport_eqn)

        self.variable_type = variable_type
        self.phase = phase
        self.is_moisture = is_moisture
        if self.variable_type != TracerVariableType.mixing_ratio:
            raise NotImplementedError('Only mixing ratio tracers are currently implemented')


class WaterVapour(ActiveTracer):

    def __init__(self, name='vapour', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_flag=True,
                 transport_eqn=TransportEquationForm.advective):
        super().__init__(f'{name}_{variable_type.name}', space, variable_type,
                         transport_flag, phase=Phases.gas, is_moisture=True)


class CloudWater(ActiveTracer):

    def __init__(self, name='cloud_liquid', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_flag=True,
                 transport_eqn=TransportEquationForm.advective):
        super().__init__(f'{name}_{variable_type.name}', space, variable_type,
                         transport_flag, phase=Phases.liquid, is_moisture=True)


class Rain(ActiveTracer):

    def __init__(self, name='rain', space='theta',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_flag=True,
                 transport_eqn=TransportEquationForm.advective):
        super().__init__(f'{name}_{variable_type.name}', space, variable_type,
                         transport_flag, phase=Phases.liquid, is_moisture=True)
