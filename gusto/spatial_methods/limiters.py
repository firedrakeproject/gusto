"""
This module contains slope limiters.
Slope limiters are used in transport schemes to enforce monotonicity. They are
generally passed as an argument to time discretisations, and should be selected
to be compatible with with :class:`FunctionSpace` of the transported field.
"""

from firedrake import (BrokenElement, Function, FunctionSpace, interval,
                       FiniteElement, TensorProductElement, Constant)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from gusto.core.kernels import LimitMidpoints, ClipZero, MeanMixingRatioWeights

import numpy as np

__all__ = ["DG1Limiter", "ThetaLimiter", "NoLimiter", "ZeroLimiter",
           "MixedFSLimiter", "MeanLimiter"]


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
        if (space.ufl_element().sobolev_space.name != 'L2'
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
        sub_elements = space.ufl_element().factor_elements
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


class ZeroLimiter(object):
    """
    A simple limiter to enforce non-negativity of a field pointwise.

    Negative values are simply clipped to be zero. There is also the option to
    project the field to another function space to enforce non-negativity there.
    """

    def __init__(self, space, clipping_space=None):
        """
        Args:
            space (:class:`FunctionSpace`): the space of the incoming field to
                clip.
            clipping_space (:class:`FunctionSpace`, optional): the space in
                which to clip the field. If not specified, the space of the
                input field is used.
        """

        self.space = space
        if clipping_space is not None:
            self.clipping_space = clipping_space
            self.map_to_clip = True
            self.field_to_clip = Function(self.clipping_space)
        else:
            self.clipping_space = space
            self.map_to_clip = False

        self._kernel = ClipZero(self.clipping_space)

    def apply(self, field):
        """
        The application of the limiter to the field.

        Args:
            field (:class:`Function`): the field to apply the limiter to.
         """

        # Obtain field in clipping space
        if self.map_to_clip:
            self.field_to_clip.interpolate(field)
            self._kernel.apply(self.field_to_clip, self.field_to_clip)
            field.interpolate(self.field_to_clip)
        else:
            self._kernel.apply(field, field)


class NoLimiter(object):
    """A blank limiter that does nothing."""

    def __init__(self):
        pass

    def apply(self, field):
        """
        The application of the blank limiter.

        Args:
            field (:class:`Function`): the field to which the limiter would be
                applied, if this was not a blank limiter.
        """
        pass


class MixedFSLimiter(object):
    """
    An object to hold a dictionary that defines limiters for transported prognostic
    variables. Different limiters may be applied to different fields and not every
    transported variable needs a defined limiter.
    """

    def __init__(self, equation, sublimiters):
        """
        Args:
            equation (:class: `PrognosticEquationSet`): the prognostic equation(s)
            sublimiters (dict): A dictionary holding limiters defined for individual prognostic variables
        Raises:
            ValueError: If a limiter is defined for a field that is not in the prognostic variable set
        """

        self.sublimiters = sublimiters
        self.field_idxs = {}

        for field, _ in sublimiters.items():
            # Check that the field is in the prognostic variable set:
            if field not in equation.field_names:
                raise ValueError(f"The limiter defined for {field} is for a field that does not exist in the equation set")
            else:
                self.field_idxs[field] = equation.field_names.index(field)

    def apply(self, fields):
        """
        Apply the individual limiters to specific prognostic variables
        """

        for field, sublimiter in self.sublimiters.items():
            field = fields.subfunctions[self.field_idxs[field]]
            sublimiter.apply(field)


class MeanLimiter(object):
    """
    A mass-preserving limiter for mixing ratios that ensures non-negativity
    by blending the mixing ratio with its associated mean field.
    The blending factor is given by the DG0 function lamda. The same lamda
    is used when there are multiple fields, for mass conservation.
    """

    def __init__(self, spaces):
        """
        Args:
            spaces: The function spaces for the DG1 mixing ratios
        Raises:
            ValueError: If the space is not appropriate for the limiter, i.e DG1
        """

        # The Mean Limiter is currently set up for mixing ratios in DG1.
        for space in spaces:
            degree = space.ufl_element().degree()
            if (space.ufl_element().sobolev_space.name != 'L2'
                or ((type(degree) is tuple and np.any([deg != 1 for deg in degree]))
                    and degree != 1)):
                raise NotImplementedError('MeanLimiter only implemented for mixing'
                                          + 'ratios in the DG1 space')

        self.space = spaces[0]
        mesh = self.space.mesh()

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
        DG0 = FunctionSpace(mesh, 'DG', 0)
        DG1 = FunctionSpace(mesh, 'DG', 1)

        self.lamda = Function(DG0)
        self.mX_field = Function(DG1_equispaced)
        self.mean_field = Function(DG0)
        self.mX_new = Function(DG1_equispaced)

        self._lamda_kernel = MeanMixingRatioWeights(DG1_equispaced)

        # Also construct kernels to clip any very small negatives
        # that arise from numerical error. These are used in the
        # mean mixing ratio augmentation limit routine.
        self._clip_DG1_field = ClipZero(DG1)
        self._clip_means_kernel = ClipZero(DG0)

    def apply(self, mX_fields, mean_fields):
        """
        Compute the limiter weights, lambda, and use these
        to combine the DG1 mixing ratio and DG0 mean field
        to ensure non-negativity.

        Args:
            mX_fields (:class:`Function`): the DG1 mixing ratios to limit.
            mean_fields (:class:`Function`): the DG0 mean field associated with
            each mX_field.
         """

        # Remove weights from previous applications
        self.lamda.interpolate(Constant(0.0))

        for i in range(len(mX_fields)):
            # Interpolate fields from DG1 to DG1 equispaced
            self.mX_field.interpolate(mX_fields[i])
            self.mean_field.interpolate(mean_fields[i])

            # Update the weights based on any negative values
            self._lamda_kernel.apply(self.lamda, self.mX_field, self.mean_field)

        # Perform blended limiting, with all mixing ratios using
        # the same lambda field to ensure conservation.
        for i in range(len(mX_fields)):
            self.mX_field.interpolate(mX_fields[i])
            self.mean_field.interpolate(mean_fields[i])

            self.mX_new.interpolate((Constant(1.0) - self.lamda)*self.mX_field + self.lamda*self.mean_field)
            mX_fields[i].interpolate(self.mX_new)
