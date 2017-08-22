from abc import ABCMeta, abstractmethod


__all__ = ["MeshGenerator"]


class MeshGenerator(object, metaclass=ABCMeta):
    """
    Base class for a mesh generator for a moving mesh method.
    """

    @abstractmethod
    def get_new_mesh(self):
        pass
