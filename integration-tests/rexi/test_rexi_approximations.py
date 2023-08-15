"""
This tests each component of the REXI approximation, and sequentially builds the
full REXI approximation. The tests can be run with the original REXI
coefficients (from 'A high-order time-parallel scheme for solving wave
propagation problems via the direct construction of an approximate
time-evolution operator') or with the coefficients given in 'An accurate and
time-paralle rational exponential integrator for hyperbolic and oscillatory
PDEs'). The choice of coefficients is specified with the flag
'original_constants'.
"""

from gusto.rexi import *
from firedrake import exp, sqrt, pi
import pytest

original_constants = False

constants = REXIConstants()  # these are mu, L and a from rexi_coefficients.py
mu = constants.mu
L = constants.L
a = constants.a

params = RexiParameters()  # these are h, M and reduce_to_half from rexi.py

def approx_e_ix(x, h, M, use_Gaussian_approx):
    """
    Approximation of e^(ix), either with Gaussians approximated as fractions
    or in the full REXI approximation (writing the sum of Gaussians and the
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


def approxGaussian(x, h):
    """
    Approximation of Gaussian basis function as a sum of complex rational
    functions.
    """
    x /= h
    if original_constants:
        sum = 0
        for t in range(0, len(a)):
            l = t-L
            # WORKS with max error 7.15344e-13
            sum += (a[t]/(1j*x + mu + 1j*l)).real
    else:
        sum = (a[0]*mu)/(x**2 + mu**2)
        for l in range(1, len(a)):
            numerator = 2*mu*a[l].real*(mu**2 + l**2 + x**2) + 2*l*a[l].imag*(mu**2 + l**2 - x**2)
            denominator = x**4 + 2*(mu**2 - l**2)*x**2 + (mu**2+l**2)**2
            sum += numerator/denominator
    return sum

def approx_exp_as_Gaussian(h, M, x):
    """
    Approximation of the exponential as a sum of (exact) Gaussian functions.
    """
    b = b_coefficients(h, M)
    sum = 0
    for m in range(-M, M+1):
        sum += b[m+M] * exactGaussian(h, x+ (m*h))
    return sum


def exactGaussian(h, x):
    """
    An exact Gaussian, for comparison with approximations. 
    """
    return exp(-(x*x)/(4.0*h*h))/sqrt(4.0*pi)

# ------------------------------------------------------------------------ #
# Test 1: This tests the first REXI approximation: the exponential as a sum of
# Gaussians. The Gaussians are exact. The test compares the linear combination
# of Gaussian functions (with b coefficients as weights) with the exact
# exponential, for scalars between 1 and 10. 
# ------------------------------------------------------------------------ #

def test_bms():
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
        exact = exp(-(x*x)/(4.0*h*h))/sqrt(4.0*pi)
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
