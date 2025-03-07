"""
This module provides kernels for performing element-wise operations.

Kernels are held in classes containing the instructions and an apply method,
which calls the kernel using a par loop. The code snippets used in the kernels
are written using loopy (https://documen.tician.de/loopy/index.html)

Kernels are contained in this module so that they can be easily imported and
tested.
"""

from firedrake import dx
from firedrake.parloops import par_loop, READ, WRITE, INC, MIN, MAX, op2
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
    mean DG0 field in the MeanMixingRatioLimiter.

    First, each cell is looped over and the minimum value is computed
    """

    def __init__(self, V):
        """
        Args:
            V (:class:`FunctionSpace`): The space of the field to be clipped.
        """
        # Using DG1-equispaced, with 4 DOFs per cell
        shapes = {'nDOFs': V.finat_element.space_dimension(),
                  'nDOFs_base': int(V.finat_element.space_dimension() / 4)}
        domain = "{{[i]: 0 <= i < {nDOFs_base}}}".format(**shapes)

        instrs = ("""
                  <float64> min_value1 = 0.0
                  <float64> min_value2 = 0.0
                  <float64> min_value = 0.0
                  <float64> new_lamda = 0.0
                  for i
                      min_value1 = fmin(mX_field[i*4], mX_field[i*4+1])
                      min_value2 = fmin(mX_field[i*4+2], mX_field[i*4+3])
                      min_value = fmin(min_value1, min_value2)
                      if min_value < 0.0
                          lamda[i] = fmax(lamda[i],-min_value/(mean_field[i] - min_value))
                      end
                  end
                  """)
        #lamda[i] = fmax(lamda[i],-min_value/(mean_field[i] - min_value))
        self._kernel = (domain, instrs)

    def apply(self, lamda, mX_field, mean_field):
        """
        Performs the par loop.

        Args:
            w (:class:`Function`): the field in which to store the weights. This
                lives in the continuous target space.
        """
        par_loop(self._kernel, dx,
                 {"lamda": (lamda, INC),
                  "mX_field": (mX_field, READ),
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

        fmin = op2.Global(1, np.finfo(float).max, dtype=float, comm=field._comm)

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

        fmax = op2.Global(1, np.finfo(float).min, dtype=float, comm=field._comm)

        op2.par_loop(self._kernel, field.dof_dset.set, fmax(MAX), field.dat(READ))

        return fmax.data[0]
