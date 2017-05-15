from __future__ import absolute_import
from six import with_metaclass
from abc import ABCMeta, abstractmethod


class MeshGenerator(with_metaclass(ABCMeta)):
    """
    Base class for a mesh generator for a moving mesh method.
    """

    @abstractmethod
    def get_new_mesh(self):
        pass
