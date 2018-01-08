"""
Some thermodynamic expressions to help declutter the code.
"""
from firedrake import exp


def theta_expr(T, p, state):
    """
    Returns an expression for dry potential temperature theta in K.

    arg: T: temperature in K.
    arg: p: pressure in Pa.
    arg: state: state class to provide parameters.
    """

    kappa = state.parameters.kappa
    p_0 = state.parameters.p_0

    return T * (p_0 / p) ** kappa


def pi_expr(rho, theta_v, state):
    """
    Returns an expression for the Exner pressure.

    arg: rho: the dry density of air in kg / m^3.
    arg: theta: the potential temperature (or the virtual
                potential temperature for wet air), in K.
    arg: state: state class to provide parameters.
    """

    kappa = state.parameters.kappa
    p_0 = state.parameters.p_0
    R_d = state.parameters.R_d

    return (rho * R_d * theta_v / p_0) ** (kappa / (1 - kappa))


def p_expr(pi, state):
    """
    Returns an expression for the pressure in Pa from the Exner Pi.

    arg: pi: the Exner pressure.
    arg: state: state class to provide parameters.
    """

    kappa = state.parameters.kappa
    p_0 = state.parameters.p_0

    return p_0 * pi ** (1 / kappa)


def T_expr(theta_v, pi, state, r_v=None):
    """
    Returns an expression for temperature T in K.

    arg: theta_v: the virtual potential temperature in K.
    arg: pi: the Exner pressure.
    arg: r_v: the mixing ratio of water vapour.
    arg: state: state class to provide parameters.
    """

    R_d = state.parameters.R_d
    R_v = state.parameters.R_v

    # if the air is wet, need to divide by (1 + r_v)
    if r_v is not None:
        return theta_v * pi / (1 + r_v * R_v / R_d)
    # in the case that r_v is None, theta_v=theta
    else:
        return theta_v * pi


def rho_expr(theta_v, pi, state):
    """
    Returns an expression for the dry density rho in kg / m^3
    from the (virtual) potential temperature and Exner pressure.

    arg: theta_v: the virtual potential temperature in K.
    arg: pi: the Exner pressure.
    arg: state: state class to provide parameters.
    """

    kappa = state.parameters.kappa
    p_0 = state.parameters.p_0
    R_d = state.parameters.R_d

    return p_0 * pi ** (1 / kappa - 1) / (R_d * theta_v)


def r_sat_expr(T, p, state):
    """
    Returns an expression from Tetens' formula for the
    saturation mixing ratio of water vapour.

    arg: T: the temperature in K.
    arg: p: the pressure in Pa.
    arg: state: state class to provide parameters.
    """

    w_sat1 = state.parameters.w_sat1
    w_sat2 = state.parameters.w_sat2
    w_sat3 = state.parameters.w_sat3
    w_sat4 = state.parameters.w_sat4
    T_0 = state.parameters.T_0

    return w_sat1 / (p * exp(w_sat2 * (T - T_0) / (T - w_sat3)) - w_sat4)


def Lv_expr(T, state):
    """
    Returns an expression for the latent heat of vaporisation of water.

    arg: T: the temperature in K.
    arg: state: state class to provide parameters.
    """

    L_v0 = state.parameters.L_v0
    T_0 = state.parameters.T_0
    c_pl = state.parameters.c_pl
    c_pv = state.parameters.c_pv

    return L_v0 - (c_pl - c_pv) * (T - T_0)


def theta_e_expr(T, p, r_v, r_t, state):
    """
    Returns an expression for the wet equivalent potential temperature in K.

    arg: T: the temperature in K.
    arg: p: the pressure in Pa.
    arg: r_v: the mixing ratio of water vapour.
    arg: r_t: the total mixing ratio of water.
    arg: state: state class to provide parameters.
    """

    R_d = state.parameters.R_d
    R_v = state.parameters.R_v
    p_0 = state.parameters.p_0
    cp = state.parameters.cp
    c_pl = state.parameters.c_pl
    Lv = Lv_expr(T, state)

    return T * (p_0 * (1 + r_v * R_v / R_d) / p) ** (R_d / (cp + c_pl * r_t)) * exp(Lv * r_v / (T * (cp + c_pl * r_t)))


def I_expr(rho, T, state, r_v=0.0, r_l=0.0):
    """
    Returns an expression for the (possibly wet) internal energy density in J.

    arg: rho: the dry density in kg / m^3.
    arg: T: the temperature in K.
    arg: r_v: the mixing ratio of water vapour.
    arg: r_l: the mixing ratio of all forms of liquid water.
    arg: state: state class to provide parameters.
    """

    cv = state.parameters.cv
    c_vv = state.parameters.c_vv
    c_pv = state.parameters.c_pv
    Lv = Lv_expr(T, state)

    return rho * (cv * T + r_v * c_vv * T + r_l * (c_pv * T - Lv))
