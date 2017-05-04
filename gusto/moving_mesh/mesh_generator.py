from __future__ import absolute_import
from abc import ABCMeta, abstractmethod


class MeshGenerator(object):
    """
    Base class for a mesh generator for a moving mesh method.

    :arg mesh: mesh for underlying simulation
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_new_mesh(self):
        pass
