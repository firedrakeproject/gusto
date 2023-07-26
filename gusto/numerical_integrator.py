import numpy as np


class NumericalIntegral(object):
    """
    A class for numerically evaluating and tabulating some 1D integral.
    Args:
        lower_bound(float): lower bound of integral
        upper_bound(float): upper bound of integral
        num_points(float): number of points to tabulate integral at

    """
    def __init__(self, lower_bound, upper_bound, num_points=500):

        # if upper_bound <= lower_bound:
        #     raise ValueError('lower_bound must be lower than upper_bound')
        self.x = np.linspace(lower_bound, upper_bound, num_points)
        self.x_double = np.linspace(lower_bound, upper_bound, 2*num_points-1)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_points = num_points
        self.tabulated = False

    def tabulate(self, expression):
        """
        Tabulate some integral expression using Simpson's rule.
        Args:
            expression (func): a function representing the integrand to be
                evaluated. should take a numpy array as an argument.
        """

        self.cumulative = np.zeros_like(self.x)
        self.interval_areas = np.zeros(len(self.x)-1)
        # Evaluate expression in advance to make use of numpy optimisation
        # We evaluate at the tabulation points and the midpoints of the intervals
        f = expression(self.x_double)

        # Just do Simpson's rule for evaluating area of each interval
        self.interval_areas = ((self.x[1:] - self.x[:-1]) / 6.0
                               * (f[2::2] + 4.0 * f[1::2] + f[:-1:2]))

        # Add the interval areas together to create cumulative integral
        for i in range(self.num_points - 1):
            self.cumulative[i+1] = self.cumulative[i] + self.interval_areas[i]

        self.tabulated = True

    def evaluate_at(self, points):
        """
        Evaluates the integral at some point using linear interpolation.
        Args:
            points (float or iter) the point value, or array of point values to
                evaluate the integral at.
        Return:
            returns the numerical approximation of the integral from lower 
            bound to point(s)
        """
        # Do linear interpolation from tabulated values
        if not self.tabulated:
            raise RuntimeError(
                'Integral must be tabulated before we can evaluate it at a point')

        return np.interp(points, self.x, self.cumulative)
