"""
This tests each component of the REXI approximation, and sequentially builds the
full REXI approximation.
"""

from gusto.rexi import *
from firedrake import exp, sqrt, pi
import pytest

params = RexiParameters()
mu = params.mu
L = params.L
a = params.a


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
        sum += b[m+M] * exactGaussian(x+ (m*h), h)
    return sum


def approxGaussian(x, h):
    """
    Approximation of Gaussian basis function as a sum of complex rational
    functions.
    """
    x /= h
    sum = 0
    for t in range(0, len(a)):
        l = t-L
        sum += (a[t]/(1j*x + mu + 1j*l)).real
    return sum


def approx_e_ix(x, h, M, use_Gaussian_approx):
    """
    Approximation of e^(ix), either with Gaussians approximated as fractions
    or with the full REXI approximation (writing the sum of Gaussians and the
    sum of rational functions as a single sum).
    """
    b = b_coefficients(h, M)
    sum = 0
    if use_Gaussian_approx:
        # this is REXI with Gaussians approximated as fractions
        for m in range(-M, M+1):
            sum += b[m+M] * approxGaussian(x+m*h, h)
    else:
        # this is the full REXI (combining the sums)
        alpha, beta, beta2 = RexiCoefficients(params)
        for n in range(len(alpha)):
            denom = (1j*x + alpha[n])
            sum += beta[n] / denom

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
        exact =  exp(1j*x)
        approx = approx_exp_as_Gaussian(h, M, x)
        assert abs(exact - approx) < 1e-14

# ------------------------------------------------------------------------ #
# Test 2: This tests the second step: the approximation of the Gaussians as a
# sum of complex fractions. It compares an exact Gaussian function with an
# approximation as a sum of rational functions. The sum of rational functions
# varies depending on the choice of 'original_constants'.
# ------------------------------------------------------------------------ #

def test_gaussian_approx():
    h = 0.2
    for x in range(10):
        exact = exactGaussian(x, h)
        approx = approxGaussian(x, h)
        assert abs(exact - approx) < 7.15344e-13

# ------------------------------------------------------------------------ #
# Test 3: This combines Tests 1 and 2 to compare the exact exponential with an 
# approximation produced using a linear combination of Gaussian functions, which
# are themselves approximated by a sum of rational functions.
# ------------------------------------------------------------------------ #

def test_exponential_approx():
    h = 0.2
    M = 64
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix(x, h, M, True)
        assert abs(exact - approx) < 2.e-11

# ------------------------------------------------------------------------ #
# Test 3: This test the full REXI appromimation, i.e. the combination of steps
# 1 and 2 into a single sum. 
# ------------------------------------------------------------------------ #

def test_rexi_exponential_approx():
    h = 0.2
    M = 64
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix(x, h, M, False)
        assert abs(exact - approx) < 2.e-11
