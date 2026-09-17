"""
This module contains slope limiters.
Slope limiters are used in transport schemes to enforce monotonicity. They are
generally passed as an argument to time discretisations, and should be selected
to be compatible with with :class:`FunctionSpace` of the transported field.
"""

from firedrake import (BrokenElement, Function, FunctionSpace, interval,
                       FiniteElement, TensorProductElement, Constant,
                       min_value, max_value)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from gusto.core.kernels import (
    LimitMidpoints, ClipZero, MeanMixingRatioWeights,
    MeanMixingRatioStencilBounds, MonotonicMeanMixingRatioWeights
)

import numpy as np

__all__ = ["DG1Limiter", "ThetaLimiter", "NoLimiter", "ZeroLimiter",
           "MixedFSLimiter", "MeanLimiter", "MonotonicMeanLimiter"]


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
            cell = mesh._base_mesh.ufl_cell().cellname
            DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
            DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
            DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        else:
            cell = mesh.ufl_cell().cellname
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
        cell = mesh._base_mesh.ufl_cell().cellname
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
            cell = mesh._base_mesh.ufl_cell().cellname
            DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
            DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
            DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        else:
            cell = mesh.ufl_cell().cellname
            DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")

        DG1_equispaced = FunctionSpace(mesh, DG1_element)
        DG0 = FunctionSpace(mesh, 'DG', 0)

        self.lamda = Function(DG0)
        self.mX_field = Function(DG1_equispaced)
        self.mean_field = Function(DG0)
        self.mX_new = Function(DG1_equispaced)

        self._lamda_kernel = MeanMixingRatioWeights(DG1_equispaced)

        # Also construct a kernels to clip any very small negatives
        # that arise from numerical error when computing the
        # mean mixing ratio.
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


class MonotonicMeanLimiter(object):
    """
    A mass-preserving limiter for mixing ratios that enforces monotonicity
    (rather than just non-negativity) by blending the transported mixing
    ratio with its associated mean field.

    Following the derivation in monotone_limiter.tex, in each cell e the
    blending weight lamda_e is chosen so that the limited field
    m*_e = (1-lamda_e)*m^{n+1}_e + lamda_e*mbar_e
    lies within the minimum and maximum values taken by the pre-transport
    field over the cell e and its facet-neighbours, e union d(e). As with
    :class:`MeanLimiter`, the same lamda field is used to blend every mixing
    ratio provided, so that mass is conserved.
    """

    def __init__(self, spaces, enforce_nonnegative=False, extruded_bounds_method='facet'):
        """
        Args:
            spaces: The function spaces for the DG1 mixing ratios
            enforce_nonnegative (bool, optional): whether to additionally
                clip small negative values from the mean field, to guard
                against numerical error in its computation. This should
                not be used if the mixing ratio may be legitimately
                negative, since monotonicity does not imply non-negativity.
                Defaults to False.
            extruded_bounds_method (str, optional): for extruded meshes, how
                to gather the min/max stencil bounds of the pre-transport
                field. Options are:
                - 'relaxed': gather bounds using a single CG1 space, so that
                  any cells sharing a vertex (including diagonal neighbours
                  across a layer and column) contribute to the bounds. This
                  gives valid, but more relaxed, monotonic bounds.
                - 'facet': gather bounds using a pair of tensor-product
                  spaces (one continuous in the horizontal and discontinuous
                  in the vertical, and vice versa), so that only the cell's
                  true facet-neighbours contribute to the bounds. This gives
                  tighter bounds, but relies on the horizontal base mesh
                  having the property that cells sharing a horizontal vertex
                  are also horizontal facet-neighbours (true e.g. for the 1D
                  meshes used to build vertical-slice extruded meshes).
                Ignored for non-extruded meshes, which always use a CG1
                space. Defaults to 'facet'.
        Raises:
            ValueError: If the space is not appropriate for the limiter, i.e DG1
            ValueError: If extruded_bounds_method is not a recognised option
        """

        # The Monotonic Mean Limiter is currently set up for mixing ratios in DG1.
        for space in spaces:
            degree = space.ufl_element().degree()
            if (space.ufl_element().sobolev_space.name != 'L2'
                or ((type(degree) is tuple and np.any([deg != 1 for deg in degree]))
                    and degree != 1)):
                raise NotImplementedError('MonotonicMeanLimiter only implemented for mixing'
                                          + 'ratios in the DG1 space')

        if extruded_bounds_method not in ['relaxed', 'facet']:
            raise ValueError("extruded_bounds_method must be either 'relaxed' or 'facet', "
                             + f"got '{extruded_bounds_method}'")

        self.space = spaces[0]
        mesh = self.space.mesh()
        self.extruded = mesh.extruded
        self.extruded_bounds_method = extruded_bounds_method

        # Create equispaced DG1 space needed for limiting
        if mesh.extruded:
            base_cell = mesh._base_mesh.ufl_cell().cellname
            DG1_hori_elt = FiniteElement("DG", base_cell, 1, variant="equispaced")
            DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
            DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        else:
            cell = mesh.ufl_cell().cellname
            DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")

        DG1_equispaced = FunctionSpace(mesh, DG1_element)
        DG0 = FunctionSpace(mesh, 'DG', 0)

        self.lamda = Function(DG0)
        self.new_field = Function(DG1_equispaced)
        self.old_field = Function(DG1_equispaced)
        self.mean_field = Function(DG0)
        self.mX_new = Function(DG1_equispaced)

        self.stencil_min_dg1 = Function(DG1_equispaced)
        self.stencil_max_dg1 = Function(DG1_equispaced)

        self._stencil_bounds_kernel = MeanMixingRatioStencilBounds(DG1_equispaced)
        self._lamda_kernel = MonotonicMeanMixingRatioWeights(DG1_equispaced)

        if mesh.extruded and extruded_bounds_method == 'facet':
            # Gather bounds separately over horizontal facet-neighbours (via a
            # space that's continuous in the horizontal, discontinuous in the
            # vertical) and vertical facet-neighbours (continuous in the
            # vertical, discontinuous in the horizontal), then combine them.
            CG1_hori_elt = FiniteElement("CG", base_cell, 1)
            CG1_vert_elt = FiniteElement("CG", interval, 1)
            horiz_neighbour_elt = TensorProductElement(CG1_hori_elt, DG1_vert_elt)
            vert_neighbour_elt = TensorProductElement(DG1_hori_elt, CG1_vert_elt)
            self.stencil_bounds_space_horiz = FunctionSpace(mesh, horiz_neighbour_elt)
            self.stencil_bounds_space_vert = FunctionSpace(mesh, vert_neighbour_elt)

            self.stencil_min_horiz = Function(self.stencil_bounds_space_horiz)
            self.stencil_max_horiz = Function(self.stencil_bounds_space_horiz)
            self.stencil_min_vert = Function(self.stencil_bounds_space_vert)
            self.stencil_max_vert = Function(self.stencil_bounds_space_vert)
            self.stencil_min_dg1_horiz = Function(DG1_equispaced)
            self.stencil_max_dg1_horiz = Function(DG1_equispaced)
        else:
            # CG1 space used to gather min/max values across cells sharing a
            # vertex. On extruded meshes, this includes diagonal neighbours,
            # giving valid but more relaxed bounds.
            CG1 = FunctionSpace(mesh, 'CG', 1)
            self.stencil_min_cg = Function(CG1)
            self.stencil_max_cg = Function(CG1)

        # Whether to additionally clip small negatives from the mean field
        # that arise from numerical error when computing it. This is kept
        # optional, since monotonicity alone does not require non-negativity.
        self.enforce_nonnegative = enforce_nonnegative
        self._clip_means_kernel = ClipZero(DG0)

    def apply(self, mX_fields, mean_fields, old_mX_fields):
        """
        Compute the limiter weights, lambda, and use these to combine the
        DG1 mixing ratio and DG0 mean field to ensure monotonicity.

        Args:
            mX_fields (list of :class:`Function`): the transported (pre-
                limited) DG1 mixing ratios to limit.
            mean_fields (list of :class:`Function`): the DG0 mean field
                associated with each mX_field.
            old_mX_fields (list of :class:`Function`): the DG1 mixing ratios
                before this step's transport, used to compute the monotonic
                bounds for each cell and its facet-neighbours.
         """

        # Remove weights from previous applications
        self.lamda.interpolate(Constant(0.0))

        if self.enforce_nonnegative:
            for mean_field in mean_fields:
                self._clip_means_kernel.apply(mean_field, mean_field)

        for i in range(len(mX_fields)):
            # Gather the min/max of the pre-transport field over each cell
            # and its facet-neighbours
            self.old_field.interpolate(old_mX_fields[i])

            if self.extruded and self.extruded_bounds_method == 'facet':
                # Gather bounds over horizontal facet-neighbours only (using
                # a space continuous in the horizontal, discontinuous in the
                # vertical)
                self.stencil_min_horiz.assign(1.0e10)
                self.stencil_max_horiz.assign(-1.0e10)
                self._stencil_bounds_kernel.apply(
                    self.stencil_min_horiz, self.stencil_max_horiz, self.old_field
                )
                self.stencil_min_dg1_horiz.interpolate(self.stencil_min_horiz)
                self.stencil_max_dg1_horiz.interpolate(self.stencil_max_horiz)

                # Gather bounds over vertical facet-neighbours only (using a
                # space continuous in the vertical, discontinuous in the
                # horizontal)
                self.stencil_min_vert.assign(1.0e10)
                self.stencil_max_vert.assign(-1.0e10)
                self._stencil_bounds_kernel.apply(
                    self.stencil_min_vert, self.stencil_max_vert, self.old_field
                )
                self.stencil_min_dg1.interpolate(self.stencil_min_vert)
                self.stencil_max_dg1.interpolate(self.stencil_max_vert)

                # Combine the two, so that the bounds are taken over the
                # cell and its true facet-neighbours (horizontal and
                # vertical) only
                self.stencil_min_dg1.interpolate(
                    min_value(self.stencil_min_dg1, self.stencil_min_dg1_horiz)
                )
                self.stencil_max_dg1.interpolate(
                    max_value(self.stencil_max_dg1, self.stencil_max_dg1_horiz)
                )
            else:
                self.stencil_min_cg.assign(1.0e10)
                self.stencil_max_cg.assign(-1.0e10)
                self._stencil_bounds_kernel.apply(
                    self.stencil_min_cg, self.stencil_max_cg, self.old_field
                )
                self.stencil_min_dg1.interpolate(self.stencil_min_cg)
                self.stencil_max_dg1.interpolate(self.stencil_max_cg)

            # Interpolate fields from DG1 to DG1 equispaced
            self.new_field.interpolate(mX_fields[i])
            self.mean_field.interpolate(mean_fields[i])

            # Update the weights based on the monotonic bounds
            self._lamda_kernel.apply(
                self.lamda, self.new_field,
                self.stencil_min_dg1, self.stencil_max_dg1, self.mean_field
            )

        # Perform blended limiting, with all mixing ratios using
        # the same lambda field to ensure conservation.
        for i in range(len(mX_fields)):
            self.new_field.interpolate(mX_fields[i])
            self.mean_field.interpolate(mean_fields[i])

            self.mX_new.interpolate((Constant(1.0) - self.lamda)*self.new_field + self.lamda*self.mean_field)
            mX_fields[i].interpolate(self.mX_new)
