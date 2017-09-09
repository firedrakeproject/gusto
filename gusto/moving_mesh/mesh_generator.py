from abc import ABCMeta, abstractmethod


__all__ = ["MeshGenerator"]


class MeshGenerator(object, metaclass=ABCMeta):
    """
    Base class for a mesh generator for a moving mesh method.

    :arg pre_meshgen_callback: optional user-supplied callback function
    that is executed each timestep before a new mesh is made
    :arg post_meshgen_callback: optional user-supplied callback function
    that is executed each timestep after a new mesh is made
    """

    def __init__(self, pre_meshgen_callback=None, post_meshgen_callback=None):
        self.pre_meshgen_fn = pre_meshgen_callback
        self.post_meshgen_fn = post_meshgen_callback

    @abstractmethod
    def get_new_mesh(self):
        pass

    def pre_meshgen_callback(self):
        if self.pre_meshgen_fn:
            self.pre_meshgen_fn()

    def post_meshgen_callback(self):
        if self.post_meshgen_fn:
            self.post_meshgen_fn()
