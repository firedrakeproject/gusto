"""
This tests each component of the REXI approximation, and sequentially builds the
full REXI approximation.
"""

from gusto.rexi import *
from firedrake import exp, sqrt, pi
import pytest


def exactGaussian(x, h):
    """
    An exact Gaussian, for comparison with approximations.
    """
    return exp(-(x*x)/(4.0*h*h))/sqrt(4.0*pi)


def approx_exp_as_Gaussian(h, M, x):
    """
    Approximation of the exponential as a sum of (exact) Gaussian functions.
    """
    b = b_coefficients(h, M)
    sum = 0
    for m in range(-M, M+1):
        sum += b[m+M] * exactGaussian(x + (m * h), h)
    return sum


def approxGaussian(x, params):
    """
    Approximation of Gaussian basis function as a sum of complex rational
    functions.
    """
    h = params.h
    consts = RexiConstants(params.coefficients)
    mu = consts.mu
    a = consts.a
    L = consts.L
    x /= h
    sum = 0
    for t in range(0, len(a)):
        l = t-L
        sum += (a[t]/(1j*x + mu + 1j*l)).real
    return sum


def approx_e_ix(x, params, approx_type):
    """
    Approximation of e^(ix), either with Gaussians approximated as fractions
    or with the full REXI/REXII approximation (writing the sum of Gaussians and the
    sum of rational functions as a single sum).
    """
    sum = 0
    if approx_type == "Gaussian":
        # this is REXI with Gaussians approximated as fractions
        h = params.h
        M = params.M
        b = b_coefficients(h, M)
        for m in range(-M, M+1):
            sum += b[m+M] * approxGaussian(x+m*h, h)
    elif approx_type == "REXI":
        # this is the full REXI (combining the sums)
        alpha, beta, beta2 = RexiCoefficients(params)
        for n in range(len(alpha)):
            denom = (1j*x + alpha[n])
            sum += beta[n] / denom
    elif approx_type == "REXII":
        # This is the REXII scheme from Caliari et al
        h = params.h
        M = params.M
        # FIX ME !!!
        mu = -5.133333333333333 + 1j*0
        alpha, c1, c2 = RexiiCoefficients(params)
        N = int((len(alpha) - 1)/2)
        for n in range(-N, N):
            numer = c1[n+N]*h*mu + c2[n+N]*(x + h*n)
            denom = (alpha[-n+N] - 1j*x) * (alpha[n+N] + 1j*x)
            sum += numer / denom
    else:
        raise ValueError(f"approx_type must be one of Gaussian, REXI or REXII but received {approx_type}")

    return sum

# ------------------------------------------------------------------------ #
# Test 1: This tests the first REXI approximation: the exponential as a sum of
# Gaussians. The Gaussians are exact. The test compares the linear combination
# of Gaussian functions (with b coefficients as weights) with the exact
# exponential, for scalars between 1 and 10.
# ------------------------------------------------------------------------ #


def test_sum_of_gaussian_approx():
    h = 0.2
    M = 64
    for x in range(10):
        exact = exp(1j * x)
        approx = approx_exp_as_Gaussian(h, M, x)
        assert abs(exact - approx) < 1e-14

# ------------------------------------------------------------------------ #
# Test 2: This tests the second step: the approximation of the Gaussians as a
# sum of complex fractions. It compares an exact Gaussian function with an
# approximation as a sum of rational functions. The sum of rational functions
# varies depending on the choice of 'original_constants'.
# ------------------------------------------------------------------------ #


@pytest.mark.parametrize("coefficients", ["Haut", "Caliari"])
def test_gaussian_approx(coefficients):
    params = RexiParameters(coefficients=coefficients)
    for x in range(10):
        exact = exactGaussian(x, params.h)
        approx = approxGaussian(x, params)
        assert abs(exact - approx) < 7.15344e-13

# ------------------------------------------------------------------------ #
# Test 3: This combines Tests 1 and 2 to compare the exact exponential with an
# approximation produced using a linear combination of Gaussian functions, which
# are themselves approximated by a sum of rational functions.
# ------------------------------------------------------------------------ #


@pytest.mark.parametrize("coefficients", ["Haut", "Caliari"])
def test_exponential_approx(coefficients):
    params = RexiParameters(coefficients=coefficients)
    h = params.h
    M = params.h
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix(x, params, True)
        assert abs(exact - approx) < 2.e-11

# ------------------------------------------------------------------------ #
# Test 3: This test the full REXI appromimation, i.e. the combination of steps
# 1 and 2 into a single sum.
# ------------------------------------------------------------------------ #


@pytest.mark.parametrize("algorithm", ["REXI_Haut", "REXI_Caliari", "REXII_Caliari"])
def test_rexi_exponential_approx(algorithm):

    match algorithm:
        case "REXI_Haut":
            params = RexiParameters(coefficients="Haut")
            approx_type = "REXI"
        case "REXI_Caliari":
            params = RexiParameters(coefficients="Caliari")
            approx_type = "REXI"
        case "REXII_Caliari":
            params = RexiParameters(coefficients="Caliari")
            approx_type = "REXII"
        case _:
            raise ValueError("Algorithm must be one of: REXI_Haut, REXI_Caliari or REXII_Caliari.")

    h = params.h
    M = params.M
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix(x, params, approx_type)
        assert abs(exact - approx) < 8.e-11
