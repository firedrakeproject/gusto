"""
Some thermodynamic expressions to help declutter the code.
"""
from firedrake import exp


__all__ = ["theta", "pi", "pi_rho", "pi_theta", "p", "T", "rho", "r_sat", "Lv", "theta_e", "internal_energy"]


def theta(parameters, T, p):
    """
    Returns an expression for dry potential temperature theta in K.

    :arg parameters: a CompressibleParameters object.
    :arg T: temperature in K.
    :arg p: pressure in Pa.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0

    return T * (p_0 / p) ** kappa


def pi(parameters, rho, theta_v):
    """
    Returns an expression for the Exner pressure.

    :arg parameters: a CompressibleParameters object.
    :arg rho: the dry density of air in kg / m^3.
    :arg theta: the potential temperature (or the virtual
                potential temperature for wet air), in K.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return (rho * R_d * theta_v / p_0) ** (kappa / (1 - kappa))


def pi_rho(parameters, rho, theta_v):
    """
    Returns an expression for the derivative of Exner pressure
    with respect to density.

    :arg parameters: a CompressibleParameters object.
    :arg rho: the dry density of air in kg / m^3.
    :arg theta: the potential temperature (or the virtual
                potential temperature for wet air), in K.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return (kappa / (1 - kappa)) * (rho * R_d * theta_v / p_0) ** (kappa / (1 - kappa)) / rho


def pi_theta(parameters, rho, theta_v):
    """
    Returns an expression for the deriavtive of Exner pressure
    with respect to potential temperature.

    :arg parameters: a CompressibleParameters object.
    :arg rho: the dry density of air in kg / m^3.
    :arg theta: the potential temperature (or the virtual
                potential temperature for wet air), in K.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return (kappa / (1 - kappa)) * (rho * R_d * theta_v / p_0) ** (kappa / (1 - kappa)) / theta_v


def p(parameters, pi):
    """
    Returns an expression for the pressure in Pa from the Exner Pi.

    :arg parameters: a CompressibleParameters object.
    :arg pi: the Exner pressure.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0

    return p_0 * pi ** (1 / kappa)


def T(parameters, theta_v, pi, r_v=None):
    """
    Returns an expression for temperature T in K.

    :arg parameters: a CompressibleParameters object.
    :arg theta_v: the virtual potential temperature in K.
    :arg pi: the Exner pressure.
    :arg r_v: the mixing ratio of water vapour.
    """

    R_d = parameters.R_d
    R_v = parameters.R_v

    # if the air is wet, need to divide by (1 + r_v)
    if r_v is not None:
        return theta_v * pi / (1 + r_v * R_v / R_d)
    # in the case that r_v is None, theta_v=theta
    else:
        return theta_v * pi


def rho(parameters, theta_v, pi):
    """
    Returns an expression for the dry density rho in kg / m^3
    from the (virtual) potential temperature and Exner pressure.

    :arg parameters: a CompressibleParameters object.
    :arg theta_v: the virtual potential temperature in K.
    :arg pi: the Exner pressure.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return p_0 * pi ** (1 / kappa - 1) / (R_d * theta_v)


def r_sat(parameters, T, p):
    """
    Returns an expression from Tetens' formula for the
    saturation mixing ratio of water vapour.

    :arg parameters: a CompressibleParameters object.
    :arg T: the temperature in K.
    :arg p: the pressure in Pa.
    """

    w_sat1 = parameters.w_sat1
    w_sat2 = parameters.w_sat2
    w_sat3 = parameters.w_sat3
    w_sat4 = parameters.w_sat4
    T_0 = parameters.T_0

    return w_sat1 / (p * exp(w_sat2 * (T - T_0) / (T - w_sat3)) - w_sat4)


def Lv(parameters, T):
    """
    Returns an expression for the latent heat of vaporisation of water.

    :arg parameters: a CompressibleParameters object.
    :arg T: the temperature in K.
    """

    L_v0 = parameters.L_v0
    T_0 = parameters.T_0
    c_pl = parameters.c_pl
    c_pv = parameters.c_pv

    return L_v0 - (c_pl - c_pv) * (T - T_0)


def theta_e(parameters, T, p, r_v, r_t):
    """
    Returns an expression for the wet equivalent potential temperature in K.

    :arg parameters: a CompressibleParameters object.
    :arg T: the temperature in K.
    :arg p: the pressure in Pa.
    :arg r_v: the mixing ratio of water vapour.
    :arg r_t: the total mixing ratio of water.
    """

    R_d = parameters.R_d
    R_v = parameters.R_v
    p_0 = parameters.p_0
    cp = parameters.cp
    c_pl = parameters.c_pl
    L_v = Lv(parameters, T)

    return T * (p_0 * (1 + r_v * R_v / R_d) / p) ** (R_d / (cp + c_pl * r_t)) * exp(L_v * r_v / (T * (cp + c_pl * r_t)))


def internal_energy(parameters, rho, T, r_v=0.0, r_l=0.0):
    """
    Returns an expression for the (possibly wet) internal energy density in J.

    :arg parameters: a CompressibleParameters object.
    :arg rho: the dry density in kg / m^3.
    :arg T: the temperature in K.
    :arg r_v: the mixing ratio of water vapour.
    :arg r_l: the mixing ratio of all forms of liquid water.
    """

    cv = parameters.cv
    c_vv = parameters.c_vv
    c_pv = parameters.c_pv
    L_v = Lv(parameters, T)

    return rho * (cv * T + r_v * c_vv * T + r_l * (c_pv * T - L_v))
