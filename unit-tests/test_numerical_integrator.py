"""
Tests the numerical integrator.
"""
from gusto import NumericalIntegral
from numpy import sin, pi
import pytest


def quadratic(x):
    return x**2


def sine(x):
    return sin(x)


@pytest.mark.parametrize("integrand_name", ["quadratic", "sine"])
def test_numerical_integrator(integrand_name):
    if integrand_name == "quadratic":
        integrand = quadratic
        upperbound = 3
        answer = 9
    elif integrand_name == "sine":
        integrand = sine
        upperbound = pi
        answer = 2
    else:
        raise ValueError(f'{integrand_name} integrand not recognised')
    numerical_integral = NumericalIntegral(0, upperbound)
    numerical_integral.tabulate(integrand)
    area = numerical_integral.evaluate_at(upperbound)
    err_tol = 1e-10
    assert abs(area-answer) < err_tol, \
        f'numerical integrator is incorrect for {integrand_name} function'
