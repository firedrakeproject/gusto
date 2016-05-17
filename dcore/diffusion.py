from __future__ import absolute_import
from abc import ABCMeta, abstractmethod


class Diffusion(object):
    """
    Base class for diffusion schemes for dcore.

    :arg state: :class:`.State` object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        self.state = state

    @abstractmethod
    def apply(self, x, x_out):
        """
        Function takes x as input, computes F(x) and returns x_out
        as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass


class NoDiffusion(Diffusion):
    """
    An non-diffusion scheme that does nothing.
    """

    def apply(self, x_in, x_out):

        x_out.assign(x_in)
