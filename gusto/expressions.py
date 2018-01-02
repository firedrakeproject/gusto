"""
Some thermodynamic expressions to help declutter the code.
"""
from gusto import *
from firedrake import exp

# First obtain all the compressible parameters
X = CompressibleParameters()
g = X.g
N = X.N
cp = X.cp
R_d = X.R_d
kappa = X.kappa
p_0 = X.p_0
cv = X.cv
c_pl = X.c_pl
c_pv = X.c_pv
c_vv = X.c_vv
R_v = X.R_v
L_v0 = X.L_v0
T_0 = X.T_0
w_sat1 = X.w_sat1
w_sat2 = X.w_sat2
w_sat3 = X.w_sat3
w_sat4 = X.w_sat4

def theta_expr(T, p):
    """
    Returns an expression for dry potential temperature theta in K.

    arg: T: temperature in K.
    arg: p: pressure in Pa.
    """

    return T * (p_0 / p) ** kappa


def pi_expr(rho, theta_v):
    """
    Returns an expression for the Exner pressure.

    arg: rho: the dry density of air in kg / m^3.
    arg: theta: the potential temperature (or the virtual
                potential temperature for wet air), in K.
    """

    return (rho * R_d * theta_v / p_0) ** (kappa / (1 - kappa))


def p_expr(pi):
    """
    Returns an expression for the pressure in Pa from the Exner Pi.

    arg: pi: the Exner pressure.
    """

    return p_0 * pi ** (1 / kappa)


def T_expr(theta_v, pi, r_v=None):
    """
    Returns an expression for temperature T in K.

    arg: theta_v: the virtual potential temperature in K.
    arg: pi: the Exner pressure.
    arg: r_v: the mixing ratio of water vapour.
    """

    # if the air is wet, need to divide by (1 + r_v)
    if r_v is not None:
        return theta_v * pi / (1 + r_v * R_v / R_d)
    # in the case that r_v is None, theta_v=theta
    else:
        return theta_v * pi


def rho_expr(theta_v, pi):
    """
    Returns an expression for the dry density rho in kg / m^3
    from the (virtual) potential temperature and Exner pressure.

    arg: theta_v: the virtual potential temperature in K.
    arg: pi: the Exner pressure.
    """

    return p_0 * pi ** (1 / kappa - 1) / (R_d * theta_v)


def r_sat_expr(T, p):
    """
    Returns an expression from Tetens' formula for the 
    saturation mixing ratio of water vapour.

    arg: T: the temperature in K.
    arg: p: the pressure in Pa.
    """

    return w_sat1 / (p * exp(w_sat2 * (T - T_0) / (T - w_sat3)) - w_sat4)


def Lv_expr(T):
    """
    Returns an expression for the latent heat of vaporisation of water.

    arg: T: the temperature in K.
    """

    return L_v0 - (c_pl - c_pv) * (T - T_0)


def theta_e_expr(T, p, r_v, r_t):
    """
    Returns an expression for the wet equivalent potential temperature in K.

    arg: T: the temperature in K.
    arg: p: the pressure in Pa.
    arg: r_v: the mixing ratio of water vapour.
    arg: r_t: the total mixing ratio of water.
    """

    Lv = Lv_expr(T)

    return T * (p_0 * (1 + r_v * R_v / R_d) / p) ** (R_d / (cp + c_pl * r_t)) * exp(Lv * r_v / (T * (cp + c_pl * r_t)))
