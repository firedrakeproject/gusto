"""Some expressions representing common thermodynamic variables."""

from firedrake import exp, ln

__all__ = ["theta", "exner_pressure", "dexner_drho", "dexner_dtheta", "p", "T",
           "rho", "r_sat", "Lv", "theta_e", "internal_energy", "RH", "e_sat",
           "r_v", "T_dew"]


def theta(parameters, T, p):
    """
    Returns an expression for dry potential temperature theta in K.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        T (:class:`ufl.Expr`): temperature in K.
        p (:class:`ufl.Expr`): pressure in Pa.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0

    return T * (p_0 / p) ** kappa


def exner_pressure(parameters, rho, theta_vd):
    """
    Returns an expression for the Exner pressure.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        rho (:class:`ufl.Expr`): the dry density of air in kg / m^3.
        theta_vd (:class:`ufl.Expr`): the potential temperature (or the virtual
            dry potential temperature for wet air), in K.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return (rho * R_d * theta_vd / p_0) ** (kappa / (1 - kappa))


def dexner_drho(parameters, rho, theta_vd):
    """
    Returns an expression for the derivative of Exner pressure w.r.t. density.

    The derivative of the Exner pressure with respect to density.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        rho (:class:`ufl.Expr`): the dry density of air in kg / m^3.
        theta_vd (:class:`ufl.Expr`): the potential temperature (or the virtual
            dry potential temperature for wet air), in K.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return (kappa / (1 - kappa)) * (rho * R_d * theta_vd / p_0) ** (kappa / (1 - kappa)) / rho


def dexner_dtheta(parameters, rho, theta_vd):
    """
    Returns an expression for the derivative of Exner pressure w.r.t. theta.

    The derivative of the Exner pressure with respect to potential temperature.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        rho (:class:`ufl.Expr`): the dry density of air in kg / m^3.
        theta_vd (:class:`ufl.Expr`): the potential temperature (or the virtual
            potential temperature for wet air), in K.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return (kappa / (1 - kappa)) * (rho * R_d * theta_vd / p_0) ** (kappa / (1 - kappa)) / theta_vd


def p(parameters, exner):
    """
    Returns an expression for the pressure in Pa from the Exner Pi.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        exner (:class:`ufl.Expr`): the Exner pressure.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0

    return p_0 * exner ** (1 / kappa)


def T(parameters, theta_vd, exner, r_v=None):
    """
    Returns an expression for temperature T in K.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        theta_vd (:class:`ufl.Expr`): virtual dry potential temperature in K.
        exner (:class:`ufl.Expr`): the Exner pressure.
        r_v (:class:`ufl.Expr`): the mixing ratio of water vapour.
    """

    R_d = parameters.R_d
    R_v = parameters.R_v

    # if the air is wet, need to divide by (1 + r_v)
    if r_v is not None:
        return theta_vd * exner / (1 + r_v * R_v / R_d)
    # in the case that r_v is None, theta_vd=theta
    else:
        return theta_vd * exner


def rho(parameters, theta_vd, exner):
    """
    Returns an expression for the dry density rho in kg / m^3

    This is computed from the dry virtual potential temperature and the Exner
    pressure.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        theta_vd (:class:`ufl.Expr`): the virtual potential temperature in K.
        exner (:class:`ufl.Expr`): the Exner pressure.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return p_0 * exner ** (1 / kappa - 1) / (R_d * theta_vd)


def r_sat(parameters, T, p):
    """
    Returns an expression for the saturation mixing ratio of water vapour.

    It is calculated from the temperature and pressure via Tetens' formula.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        T (:class:`ufl.Expr`): the temperature in K.
        p (:class:`ufl.Expr`): the pressure in Pa.
    """

    epsilon = parameters.R_d / parameters.R_v
    esat = e_sat(parameters, T)

    return esat * epsilon / (p - esat)


def Lv(parameters, T):
    """
    Returns an expression for the latent heat of vaporisation of water.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        T (:class:`ufl.Expr`): the temperature in K.
    """

    L_v0 = parameters.L_v0
    T_0 = parameters.T_0
    c_pl = parameters.c_pl
    c_pv = parameters.c_pv

    return L_v0 - (c_pl - c_pv) * (T - T_0)


def theta_e(parameters, T, p, r_v, r_t):
    """
    Returns an expression for the wet equivalent potential temperature in K.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        T (:class:`ufl.Expr`): the temperature in K.
        p (:class:`ufl.Expr`): the pressure in Pa.
        r_v (:class:`ufl.Expr`): the mixing ratio of water vapour.
        r_t (:class:`ufl.Expr`): the total mixing ratio of water.
    """

    R_d = parameters.R_d
    R_v = parameters.R_v
    p_0 = parameters.p_0
    cp = parameters.cp
    c_pl = parameters.c_pl
    L_v = Lv(parameters, T)
    H = RH(parameters, r_v, T, p)

    return T * (p_0 * (1 + r_v * R_v / R_d) / p) ** (R_d / (cp + c_pl * r_t)) * exp(L_v * r_v / (T * (cp + c_pl * r_t))) * H ** (-r_v * R_v / (cp + c_pl * r_t))


def internal_energy(parameters, rho, T, r_v=0.0, r_l=0.0):
    """
    Returns an expression for the (possibly wet) internal energy density in J.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        rho (:class:`ufl.Expr`): the dry density in kg / m^3.
        T (:class:`ufl.Expr`): the temperature in K.
        r_v (:class:`ufl.Expr`): the mixing ratio of water vapour.
        r_l (:class:`ufl.Expr`): the mixing ratio of all forms of liquid water.
    """

    cv = parameters.cv
    c_vv = parameters.c_vv
    c_pv = parameters.c_pv
    L_v = Lv(parameters, T)

    return rho * (cv * T + r_v * c_vv * T + r_l * (c_pv * T - L_v))


def RH(parameters, r_v, T, p):
    """
    Returns an expression for the relative humidity.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        r_v (:class:`ufl.Expr`): the mixing ratio of water vapour.
        T (:class:`ufl.Expr`): the temperature in K.
        p (:class:`ufl.Expr`): the pressure in Pa.
    """

    epsilon = parameters.R_d / parameters.R_v
    rsat = r_sat(parameters, T, p)

    return r_v * (1 + rsat / epsilon) / (rsat * (1 + r_v / epsilon))


def e_sat(parameters, T):
    """
    Returns an expression for the saturated partial pressure of water vapour.

    It is calculated as a function of T, based on Tetens' formula.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        T (:class:`ufl.Expr`): the temperature in K.
    """

    w_sat2 = parameters.w_sat2
    w_sat3 = parameters.w_sat3
    w_sat4 = parameters.w_sat4
    T_0 = parameters.T_0

    return w_sat4 * exp(-w_sat2 * (T - T_0) / (T - w_sat3))


def e(parameters, p, r_v):
    """
    Returns an expression for the partial pressure of water vapour.

    It is calculated from the total pressure and the water vapour mixing ratio.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        p (:class:`ufl.Expr`): the pressure in Pa.
        r_v (:class:`ufl.Expr`): the mixing ratio of water vapour.
    """

    epsilon = parameters.R_d / parameters.R_v

    return p * r_v / (epsilon + r_v)


def r_v(parameters, H, T, p):
    """
    Returns an expression for the mixing ratio of water vapour.

    It is calculated from the relative humidity, pressure and temperature.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        H (:class:`ufl.Expr`): the relative humidity (as a decimal).
        T (:class:`ufl.Expr`): the temperature in K.
        p (:class:`ufl.Expr`): the pressure in Pa.
    """

    epsilon = parameters.R_d / parameters.R_v
    rsat = r_sat(parameters, T, p)

    return H * rsat / (1 + (1 - H) * rsat / epsilon)


# TODO: this seems incorrect!
def T_dew(parameters, p, r_v):
    """
    Returns an expression for the dewpoint temperature in K.

    It is calculated as a function of pressure and the water vapour mixing ratio.

    Args:
        parameters (:class:`CompressibleParameters`): parameters representing
            the physical constants describing the fluid.
        p (:class:`ufl.Expr`): the pressure in Pa.
        r_v (:class:`ufl.Expr`): the water vapour mixing ratio.
    """

    R_d = parameters.R_d
    R_v = parameters.R_v
    T_0 = parameters.T_0
    e = p * r_v / (r_v + R_d / R_v)

    return 243.5 / ((17.67 / ln(e / 611.2)) - 1) + T_0
