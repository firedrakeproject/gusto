"""
This module provides kernels for performing element-wise operations.

Kernels are held in classes containing the instructions and an apply method,
which calls the kernel using a par loop. The code snippets used in the kernels
are written using loopy (https://documen.tician.de/loopy/index.html)

Kernels are contained in this module so that they can be easily imported and
tested.
"""

from firedrake import dx
from firedrake.parloops import par_loop, READ, WRITE


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

        instrs = ("""
                  <float64> max_value = 0.0
                  <float64> min_value = 0.0
                  for i
                      for j
                          field_hat[i*3+j] = field_DG1[i*2+j]
                      end
                      max_value = fmax(field_DG1[i*2], field_DG1[i*2+1])
                      min_value = fmin(field_DG1[i*2], field_DG1[i*2+1])
                      if field_old[i*3+2] > max_value
                          field_hat[i*3+2] = 0.5 * (field_DG1[i*2] + field_DG1[i*2+1])
                      elif field_old[i*3+2] < min_value
                          field_hat[i*3+2] = 0.5 * (field_DG1[i*2] + field_DG1[i*2+1])
                      else
                          field_hat[i*3+2] = field_old[i*3+2]
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
                  "field_old": (field_old, READ)},
                 is_loopy_kernel=True)


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
                  "field_in": (field_in, READ)},
                 is_loopy_kernel=True)
