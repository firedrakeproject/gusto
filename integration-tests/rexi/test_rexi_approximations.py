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
    b = b_coefficients(h, M)

    sum = 0
    if use_Gaussian_approx:
        # this is REXI with Gaussians approximated as fractions
        for m in range(-M, M+1):
            sum += b[m+M] * approxGaussian(x+m*h, h)
    else:
        # this is the full REXI (testing step 3)
        alpha, beta, beta2 = RexiCoefficients(params)
        for n in range(len(alpha)):
            denom = (1j*x + alpha[n]);
            sum += beta[n] / denom

    return sum


def approxGaussian(x, h):
    """
    evaluate approximation of Gaussian basis function
    with sum of complex rational functions
    """
    if original_constants:
        x /= h

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
    b = b_coefficients(h, M)
    sum = 0
    for m in range(-M, M+1):
        sum += b[m+M] * exactGaussian(h, x+ (m*h))
    return sum


def exactGaussian(h, x):
    return exp(-(x*x)/(4.0*h*h))/sqrt(4.0*pi)


# This tests the first REXI approximation: the exponential as a sum of Gaussians
# Here the Gaussians are exact, so we are really testing the b coefficients.
def test_bms():
    h = 0.2
    M = 64
    for x in range(10):
        exact =  exp(1j*x)
        approx = approx_exp_as_Gaussian(h, M, x)
        assert abs(exact - approx) < 1e-14


# This checks the approximation of the Gaussian as a sum of complex fractions
def test_gaussian_approx():
    h = 0.2
    for x in range(10):
        exact = exp(-(x*x)/(4.0*h*h))/sqrt(4.0*pi)
        approx = approxGaussian(x, h)
        assert abs(exact - approx) < 7.15344e-13


# This checks the difference between e^(ix) and REXI where b is multiplied by
# the Gaussians, approximated as fractions.
def test_exponential_approx():
    h = 0.2
    M = 64
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix(x, h, M, True)
        assert abs(exact - approx) < 2.e-11


# This checks the full REXI approximation (combination of steps 1 and 2 in step
# 3 to make beta).
def test_rexi_exponential_approx():
    h = 0.2
    M = 64
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix(x, h, M, False)
        assert abs(exact - approx) < 2.e-11
