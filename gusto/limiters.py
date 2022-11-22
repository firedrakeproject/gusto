"""
This module contains slope limiters.
Slope limiters are used in transport schemes to enforce monotonicity. They are
generally passed as an argument to time discretisations, and should be selected
to be compatible with with :class:`FunctionSpace` of the transported field.
"""

from firedrake import (BrokenElement, Function, FunctionSpace, interval,
                       FiniteElement, TensorProductElement)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from firedrake.functionspaceimpl import IndexedFunctionSpace
from gusto.kernels import LimitMidpoints

import numpy as np

__all__ = ["DG1Limiter", "ThetaLimiter", "NoLimiter"]


class DG1Limiter(object):
    """
    A vertex-based limiter for the degree 1 discontinuous Galerkin space.

    A vertex based limiter for fields in the DG1 space. This wraps around the
    vertex-based limiter implemented in Firedrake, but ensures that this is done
    in the space using the appropriate "equispaced" elements.
    """

    def __init__(self, space, subspace=None):
        """
        Args:
            space (:class:`FunctionSpace`): the space in which the transported
                variables lies. It should be the DG1 space, or a mixed function
                space containing the DG1 space.
             subspace (int, optional): specifies that the limiter works on this
                component of a :class:`MixedFunctionSpace`.

        Raises:
            ValueError: If the space is not appropriate for the limiter.
        """

        self.space = space    # can be a mixed space
        self.subspace = subspace

        mesh = space.mesh()

        # check that space is DG1
        degree = space.ufl_element().degree()
        if (space.ufl_element().sobolev_space().name != 'L2'
            or ((type(degree) is tuple and np.any([deg != 1 for deg in degree]))
                and degree != 1)):
            raise ValueError('DG1 limiter can only be applied to DG1 space')

        # Create equispaced DG1 space needed for limiting
        if space.extruded:
            cell = mesh._base_mesh.ufl_cell().cellname()
            DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
            DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
            DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        else:
            cell = mesh.ufl_cell().cellname()
            DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")

        DG1_equispaced = FunctionSpace(mesh, DG1_element)

        self.vertex_limiter = VertexBasedLimiter(DG1_equispaced)
        self.field_equispaced = Function(DG1_equispaced)

    def apply(self, field):
        """
        The application of the limiter to the field.

        Args:
            field (:class:`Function`): the field to apply the limiter to.

        Raises:
             AssertionError: If the field is not in the correct space.
         """

        # Obtain field in equispaced DG space
        if self.subspace is not None:
            self.field_equispaced.interpolate(field.sub(self.subspace))
        else:
            self.field_equispaced.interpolate(field)
        # Use vertex based limiter on DG1 field
        self.vertex_limiter.apply(self.field_equispaced)
        # Return to original space
        if self.subspace is not None:
            field.sub(self.subspace).interpolate(self.field_equispaced)
        else:
            field.interpolate(self.field_equispaced)


class ThetaLimiter(object):
    """
    A vertex-based limiter for the degree 1 temperature space.
    A vertex based limiter for fields in the DG1xCG2 space, i.e. temperature
    variables in the next-to-lowest order set of spaces. This acts like the
    vertex-based limiter implemented in Firedrake, but in addition corrects
    the central nodes to prevent new maxima or minima forming.
    """

    def __init__(self, space):
        """
        Args:
            space (:class:`FunctionSpace`): the space in which the transported
                variables lies. It should be a form of the DG1xCG2 space.
        Raises:
            ValueError: If the mesh is not extruded.
            ValueError: If the space is not appropriate for the limiter.
        """
        if not space.extruded:
            raise ValueError('The Theta Limiter can only be used on an extruded mesh')

        # check that horizontal degree is 1 and vertical degree is 2
        sub_elements = space.ufl_element().sub_elements()
        if (sub_elements[0].family() not in ['Discontinuous Lagrange', 'DQ']
                or sub_elements[1].family() != 'Lagrange'
                or space.ufl_element().degree() != (1, 2)):
            raise ValueError('Theta Limiter should only be used with the DG1xCG2 space')

        # Transport will happen in broken form of Vtheta
        mesh = space.mesh()
        self.Vt_brok = FunctionSpace(mesh, BrokenElement(space.ufl_element()))

        # Create equispaced DG1 space needed for limiting
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        CG2_vert_elt = FiniteElement("CG", interval, 2)
        DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        Vt_element = TensorProductElement(DG1_hori_elt, CG2_vert_elt)
        DG1_equispaced = FunctionSpace(mesh, DG1_element)
        Vt_equispaced = FunctionSpace(mesh, Vt_element)
        Vt_brok_equispaced = FunctionSpace(mesh, BrokenElement(Vt_equispaced.ufl_element()))

        self.vertex_limiter = VertexBasedLimiter(DG1_equispaced)
        self.field_hat = Function(Vt_brok_equispaced)
        self.field_old = Function(Vt_brok_equispaced)
        self.field_DG1 = Function(DG1_equispaced)

        self._limit_midpoints_kernel = LimitMidpoints(Vt_brok_equispaced)

    def apply(self, field):
        """
        The application of the limiter to the field.
        Args:
            field (:class:`Function`): the field to apply the limiter to.
        Raises:
            AssertionError: If the field is not in the broken form of the
                :class:`FunctionSpace` that the :class:`ThetaLimiter` was
                initialised with.
        """
        assert field.function_space() == self.Vt_brok, \
            "Given field does not belong to this object's function space"

        # Obtain field in equispaced DG space and save original field
        self.field_old.interpolate(field)
        self.field_DG1.interpolate(field)
        # Use vertex based limiter on DG1 field
        self.vertex_limiter.apply(self.field_DG1)
        # Limit midpoints in fully equispaced Vt space
        self._limit_midpoints_kernel.apply(self.field_hat, self.field_DG1, self.field_old)
        # Return to original space
        field.interpolate(self.field_hat)


class NoLimiter(object):
    """A blank limiter that does nothing."""

    def __init__(self):
        """Initialise the blank limiter."""
        pass

    def apply(self, field):
        """
        The application of the blank limiter.
        Args:
            field (:class:`Function`): the field to which the limiter would be
                applied, if this was not a blank limiter.
        """
        pass
