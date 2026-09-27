"""
This module provides kernels for performing element-wise operations.

Kernels are held in classes containing the instructions and an apply method,
which calls the kernel using a par loop. The code snippets used in the kernels
are written using loopy (https://documen.tician.de/loopy/index.html)

Kernels are contained in this module so that they can be easily imported and
tested.
"""

from firedrake import dx
from firedrake.parloops import par_loop, READ, WRITE, RW, MIN, MAX, op2
import numpy as np


class LimitMidpoints():
    """
    Limits the vertical midpoint values for the degree 1 temperature space.

    A kernel that copies the vertex values back from the DG1 space to a broken,
    equispaced temperature space, while taking the midpoint values from the
    original field. This checks that the midpoint values are within the minimum
    and maximum at the adjacent vertices. If outside of the minimu and maximum,
    correct the values to be the average.
    """

    def __init__(self, Vt_brok):
        """
        Args:
            Vt_brok (:class:`FunctionSpace`): The broken temperature space,
                which is the space of the outputted field. The horizontal base
                element must use the equispaced variant of DG1, while the
                vertical uses CG2 (before the space has been broken).
        """
        shapes = {'nDOFs': Vt_brok.finat_element.space_dimension(),
                  'nDOFs_base': int(Vt_brok.finat_element.space_dimension() / 3)}
        domain = "{{[i,j]: 0 <= i < {nDOFs_base} and 0 <= j < 2}}".format(**shapes)
        # field_hat is in the broken theta space, assume DoFs are ordered
        # (0,1,2) in the vertical direction
        instrs = ("""
                  <float64> max_value = 0.0
                  <float64> min_value = 0.0
                  for i
                      for j
                          field_hat[i*3+2*j] = field_DG1[i*2+j]
                      end
                      max_value = fmax(field_DG1[i*2], field_DG1[i*2+1])
                      min_value = fmin(field_DG1[i*2], field_DG1[i*2+1])
                      if field_old[i*3+1] > max_value
                          field_hat[i*3+1] = 0.5 * (field_DG1[i*2] + field_DG1[i*2+1])
                      elif field_old[i*3+1] < min_value
                          field_hat[i*3+1] = 0.5 * (field_DG1[i*2] + field_DG1[i*2+1])
                      else
                          field_hat[i*3+1] = field_old[i*3+1]
                      end
                  end
                  """)

        self._kernel = (domain, instrs)

    def apply(self, field_hat, field_DG1, field_old):
        """
        Performs the par loop.

        Args:
            field_hat (:class:`Function`): The field to write to in the broken
                temperature :class:`FunctionSpace`.
            field_DG1 (:class:`Function`): A field in the equispaced DG1
                :class:`FunctionSpace` space whose vertex values have already
                been limited.
            field_old (:class:`Function`): The original unlimited field in the
                broken temperature :class:`FunctionSpace`.
        """
        par_loop(self._kernel, dx,
                 {"field_hat": (field_hat, WRITE),
                  "field_DG1": (field_DG1, READ),
                  "field_old": (field_old, READ)})


class ClipZero():
    """Clips any negative field values to be zero."""

    def __init__(self, V):
        """
        Args:
            V (:class:`FunctionSpace`): The space of the field to be clipped.
        """
        shapes = {'nDOFs': V.finat_element.space_dimension()}
        domain = "{{[i]: 0 <= i < {nDOFs}}}".format(**shapes)

        instrs = ("""
                  for i
                      if field_in[i] < 0.0
                          field[i] = 0.0
                      else
                          field[i] = field_in[i]
                      end
                  end
                  """)

        self._kernel = (domain, instrs)

    def apply(self, field, field_in):
        """
        Performs the par loop.

        Args:
            field (:class:`Function`): The field to be written to.
            field_in (:class:`Function`): The field to be clipped.
        """
        par_loop(self._kernel, dx,
                 {"field": (field, WRITE),
                  "field_in": (field_in, READ)})


class MeanMixingRatioWeights():
    """
    Finds the lambda values for blending a mixing ratio and its
    mean DG0 field in the MeanLimiter.

    The minimum value in each cell is identified.
    If the value is negative, then a lamda weight is computed
    that will ensure non-negativity in the limiting step.
    """

    def __init__(self, V_DG1):
        """
        Args:
            V (:class:`FunctionSpace`): The space of the field for the mean
            mixing ratio, which should be DG0.
        """

        shapes = {'nDOFs_DG1': V_DG1.finat_element.space_dimension()}
        domain = "{{[i]: 0 <= i < {nDOFs_DG1}}}".format(**shapes)

        instrs = ("""
                  <float64> min_value = 0.0

                  for i
                      min_value = fmin(min_value, mX_field[i])
                  end

                  if min_value < 0.0
                    lamda[0] = fmax(lamda[0],-min_value/(mean_field[0] - min_value))
                  end

                  """)

        self._kernel = (domain, instrs)

    def apply(self, lamda, mX_field, mean_field):
        """
        Performs the par loop.

        Args:
            w (:class:`Function`): the field in which to store the weights. This
                lives in the continuous target space.
        """
        par_loop(self._kernel, dx,
                 {"lamda": (lamda, RW),
                  "mX_field": (mX_field, READ),
                  "mean_field": (mean_field, READ)})


class MeanMixingRatioStencilBounds():
    """
    Gathers, at each vertex of a DG1 field, the minimum and maximum value
    taken by the field over all cells sharing that vertex.

    This is used by the :class:`MonotonicMeanLimiter` to find the minimum and
    maximum value of the pre-transported field over each cell and its
    facet-neighbours (since, for a 1D mesh, cells sharing a vertex are exactly
    the facet-neighbours of a cell). The result is stored in a continuous
    (CG1) field, so that the bounds are automatically shared between
    neighbouring cells.
    """

    def __init__(self, V_DG1):
        """
        Args:
            V_DG1 (:class:`FunctionSpace`): The (equispaced) DG1 space of the
                field whose stencil bounds are to be computed.
        """

        shapes = {'nDOFs': V_DG1.finat_element.space_dimension()}
        domain = "{{[i]: 0 <= i < {nDOFs}}}".format(**shapes)

        instrs = ("""
                  for i
                      stencil_max[i] = fmax(stencil_max[i], field[i])
                      stencil_min[i] = fmin(stencil_min[i], field[i])
                  end
                  """)

        self._kernel = (domain, instrs)

    def apply(self, stencil_min, stencil_max, field):
        """
        Performs the par loop.

        Args:
            stencil_min (:class:`Function`): the CG1 field in which to
                accumulate the minimum value at each vertex. Should be reset
                to a large value before calling this.
            stencil_max (:class:`Function`): the CG1 field in which to
                accumulate the maximum value at each vertex. Should be reset
                to a very negative value before calling this.
            field (:class:`Function`): the (equispaced) DG1 field to find the
                bounds of.
        """
        par_loop(self._kernel, dx,
                 {"stencil_min": (stencil_min, MIN),
                  "stencil_max": (stencil_max, MAX),
                  "field": (field, READ)})


class MonotonicMeanMixingRatioWeights():
    """
    Finds the lambda values for blending a mixing ratio and its mean DG0
    field in the :class:`MonotonicMeanLimiter`.

    Unlike :class:`MeanMixingRatioWeights` (which only enforces
    non-negativity), this enforces that the transported field in each cell
    lies within the minimum and maximum of the pre-transport field over that
    cell and its facet-neighbours, following the derivation in
    monotone_limiter.tex.
    """

    def __init__(self, V_DG1):
        """
        Args:
            V_DG1 (:class:`FunctionSpace`): The (equispaced) DG1 space of the
                mixing ratio field.
        """

        shapes = {'nDOFs': V_DG1.finat_element.space_dimension()}
        domain = "{{[i]: 0 <= i < {nDOFs}}}".format(**shapes)

        instrs = ("""
                  <float64> eps = 1.0e-12
                  <float64> new_min = 1.0e10
                  <float64> new_max = -1.0e10
                  <float64> old_min = 1.0e10
                  <float64> old_max = -1.0e10
                  <float64> lamda_min = 0.0
                  <float64> lamda_max = 0.0

                  for i
                      new_min = fmin(new_min, new_field[i])
                      new_max = fmax(new_max, new_field[i])
                      old_min = fmin(old_min, stencil_min[i])
                      old_max = fmax(old_max, stencil_max[i])
                  end

                  # Note: within each guarded branch below, the mean is
                  # assumed to lie within [old_min, old_max], so the
                  # denominator is guaranteed to already be non-negative
                  # (e.g. mean_field - new_min > old_min - new_min > 0 when
                  # new_min < old_min). This means only a small positive
                  # eps is needed to guard against a zero denominator; no
                  # sign-dependent epssign offset (as used elsewhere for
                  # unguarded divisions) is required here.
                  if new_min < old_min
                      lamda_min = fmin(fmax((old_min - new_min)/(mean_field[0] - new_min + eps), 0.0), 1.0)
                  end

                  if new_max > old_max
                      lamda_max = fmin(fmax((new_max - old_max)/(new_max - mean_field[0] + eps), 0.0), 1.0)
                  end

                  lamda[0] = fmax(lamda[0], fmax(lamda_min, lamda_max))
                  """)

        self._kernel = (domain, instrs)

    def apply(self, lamda, new_field, stencil_min, stencil_max, mean_field):
        """
        Performs the par loop.

        Args:
            lamda (:class:`Function`): the DG0 field in which to accumulate
                the blending weights.
            new_field (:class:`Function`): the (equispaced) DG1 pre-limited,
                post-transport mixing ratio field.
            stencil_min (:class:`Function`): the (equispaced) DG1
                representation of the minimum of the pre-transport field over
                each cell and its facet-neighbours.
            stencil_max (:class:`Function`): the (equispaced) DG1
                representation of the maximum of the pre-transport field over
                each cell and its facet-neighbours.
            mean_field (:class:`Function`): the DG0 mean mixing ratio field.
        """
        par_loop(self._kernel, dx,
                 {"lamda": (lamda, RW),
                  "new_field": (new_field, READ),
                  "stencil_min": (stencil_min, READ),
                  "stencil_max": (stencil_max, READ),
                  "mean_field": (mean_field, READ)})


class MinKernel():
    """Finds the minimum DoF value of a field."""

    def __init__(self):

        self._kernel = op2.Kernel("""
            static void minify(double *a, double *b) {
                a[0] = a[0] > b[0] ? b[0] : a[0];
            }
            """, "minify")

    def apply(self, field):
        """
        Performs the par loop.

        Args:
            field (:class:`Function`): The field to take the minimum of.

        Returns:
            The minimum DoF value of the field.
        """

        fmin = op2.Global(1, np.finfo(float).max, dtype=float, comm=field.comm)

        op2.par_loop(self._kernel, field.dof_dset.set, fmin(MIN), field.dat(READ))

        return fmin.data[0]


class MaxKernel():
    """Finds the maximum DoF value of a field."""

    def __init__(self):

        self._kernel = op2.Kernel("""
            static void maxify(double *a, double *b) {
                a[0] = a[0] < b[0] ? b[0] : a[0];
            }
            """, "maxify")

    def apply(self, field):
        """
        Performs the par loop.

        Args:
            field (:class:`Function`): The field to take the maximum of.

        Returns:
            The maximum DoF value of the field.
        """

        fmax = op2.Global(1, np.finfo(float).min, dtype=float, comm=field.comm)

        op2.par_loop(self._kernel, field.dof_dset.set, fmax(MAX), field.dat(READ))

        return fmax.data[0]
