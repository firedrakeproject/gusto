"""
This module provides kernels for performing element-wise operations.

Kernels are held in classes containing the instructions and an apply method,
which calls the kernel using a par_loop. The code snippets used in the kernels
are written using loopy: https://documen.tician.de/loopy/index.html

Kernels are contained in this module so that they can be easily imported and
tested.
"""

import numpy as np
from firedrake import dx
from firedrake.parloops import par_loop, READ, INC, WRITE
from pyop2 import ON_TOP, ON_BOTTOM


class GaussianElimination(object):
    """
    Performs local Gaussian elimination for the :class:`BoundaryRecoverer`.

    The kernel is used to improve the accuracy of the recovery process into DG1
    by performing local extrapolation at the domain boundaries. The kernel
    should be applied after an initial recovery process, which has reduced
    accuracy at the domain boundaries.

    The inaccurate initial recovery can be treated as an accurate recovery
    process at an alternative ("effective") set of coordinates. To correct this,
    the original field is expanded using a Taylor series in each cell, using the
    "effective" coordinates. This kernel performs this expansion and uses a
    Gaussian elimination process to obtain the coefficents in the Taylor
    expansion. These coefficients are then used with the actual coordinates of
    the cell's vertices to extrapolate and obtain a more accurate field on the
    domain boundaries.
    """

    def __init__(self, DG1):
        """
        Args:
            DG1 (:class:`FunctionSpace`): The equispaced DG1 function space.
        """
        shapes = {"nDOFs": DG1.finat_element.space_dimension(),
                  "dim": DG1.mesh().topological_dimension()}

        # EFF_COORDS are the effective coordinates
        # ACT_COORDS are the actual coordinates
        # DG1_OLD is the original field
        # DG1 is the target field
        # NUM_EXT is the field containing number of exterior nodes

        # In 1D EFF_COORDS and ACT_COORDS have only a single index
        # We can't generalise the expression without causing an error
        # So here we write special bits of code for the 1D case vs multi-dimensional
        if shapes['dim'] == 1:
            eff_coord_expr = (
                """
                A[i,0] = 1.0
                A[i,1] = EFF_COORDS[i]
                """)
            act_coord_expr = (
                """
                DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii]
                """)
        else:
            eff_coord_expr = (
                """
                A[i,0] = 1.0
                A[i,1] = EFF_COORDS[i,0]
                if {nDOFs} == 3
                    A[i,2] = EFF_COORDS[i,1]
                elif {nDOFs} == 4
                    A[i,2] = EFF_COORDS[i,1]
                    A[i,3] = EFF_COORDS[i,0]*EFF_COORDS[i,1]
                elif {nDOFs} == 6
                """
                # N.B we use {nDOFs} - 1 to access the z component in 3D cases
                # Otherwise loopy tries to search for this component in 2D cases, raising an error
                """
                    A[i,2] = EFF_COORDS[i,1]
                    A[i,3] = EFF_COORDS[i,{dim}-1]
                    A[i,4] = EFF_COORDS[i,0]*EFF_COORDS[i,{dim}-1]
                    A[i,5] = EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                elif {nDOFs} == 8
                    A[i,2] = EFF_COORDS[i,1]
                    A[i,3] = EFF_COORDS[i,0]*EFF_COORDS[i,1]
                    A[i,4] = EFF_COORDS[i,{dim}-1]
                    A[i,5] = EFF_COORDS[i,0]*EFF_COORDS[i,{dim}-1]
                    A[i,6] = EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                    A[i,7] = EFF_COORDS[i,0]*EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                end
                """).format(**shapes)

            act_coord_expr = (
                """
                if {nDOFs} == 2
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0]
                elif {nDOFs} == 3
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1]
                elif {nDOFs} == 4
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1]
                elif {nDOFs} == 6
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,{dim}-1] + a[4]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,{dim}-1] + a[5]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1]
                elif {nDOFs} == 8
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1] + a[4]*ACT_COORDS[iiii,{dim}-1] + a[5]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,{dim}-1] + a[6]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1] + a[7]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1]
                end
                """).format(**shapes)

        shapes['act_coord_expr'] = act_coord_expr
        shapes['eff_coord_expr'] = eff_coord_expr

        domain = (
            """
            {{[i, ii_loop, jj_loop, kk, ll_loop, mm, iii_loop, kkk_loop, iiii]:
            0 <= i < {nDOFs} and 0 <= ii_loop < {nDOFs} and
            0 <= jj_loop < {nDOFs} and 0 <= kk < {nDOFs} and
            0 <= ll_loop < {nDOFs} and 0 <= mm < {nDOFs} and
            0 <= iii_loop < {nDOFs} and 0 <= kkk_loop < {nDOFs} and
            0 <= iiii < {nDOFs}}}
            """).format(**shapes)

        instrs = (
            """
            <int> ii = 0
            <int> jj = 0
            <int> ll = 0
            <int> iii = 0
            <int> jjj = 0
            <int> i_max = 0
            <float64> A_max = 0.0
            <float64> temp_f = 0.0
            <float64> temp_A = 0.0
            <float64> c = 0.0
            <float64> f[{nDOFs}] = 0.0
            <float64> a[{nDOFs}] = 0.0
            <float64> A[{nDOFs},{nDOFs}] = 0.0
            """
            # We are aiming to find the vector a that solves A*a = f, for matrix A and vector f.
            # This is done by performing row operations (swapping and scaling) to obtain A in upper diagonal form.
            # N.B. several for loops must be executed in numerical order (loopy does not necessarily do this).
            # For these loops we must manually iterate the index.
            """
            if NUM_EXT[0] > 0.0
            """
            # only do Gaussian elimination for elements with effective coordinates
            """
                for i
            """
            # fill f with the original field values and A with the effective coordinate values
            """
                    f[i] = DG1_OLD[i]
                    a[i] = 0.0
                    {eff_coord_expr}
                end
            """
            # now loop through rows/columns of A
            """
                for ii_loop
                    A_max = abs(A[ii,ii])
                    i_max = ii
            """
            # loop to find the largest value in the ii-th column
            # set i_max as the index of the row with this largest value.
            """
                    jj = ii + 1
                    for jj_loop
                        if jj < {nDOFs}
                            if abs(A[jj,ii]) > A_max
                                i_max = jj
                            end
                            A_max = fmax(A_max, abs(A[jj,ii]))
                        end
                        jj = jj + 1
                    end
            """
            # if the max value in the ith column isn't in the ii-th row, we must swap the rows
            """
                    if i_max != ii
            """
            # swap the elements of f
            """
                        temp_f = f[ii]  {{id=set_temp_f}}
                        f[ii] = f[i_max]  {{id=set_f_imax, dep=set_temp_f}}
                        f[i_max] = temp_f  {{id=set_f_ii, dep=set_f_imax}}
            """
            # swap the elements of A
            # N.B. kk runs from ii to (nDOFs-1) as elements below diagonal should be 0
            """
                        for kk
                            if kk > ii - 1
                                temp_A = A[ii,kk]  {{id=set_temp_A}}
                                A[ii, kk] = A[i_max, kk]  {{id=set_A_ii, dep=set_temp_A}}
                                A[i_max, kk] = temp_A  {{id=set_A_imax, dep=set_A_ii}}
                            end
                        end
                    end
            """
            # scale the rows below the ith row
            """
                    ll = ii + 1
                    for ll_loop
                        if ll > ii
                            if ll < {nDOFs}
            """
            # find scaling factor
            """
                                c = - A[ll,ii] / A[ii,ii]
                                for mm
                                    A[ll, mm] = A[ll, mm] + c * A[ii,mm]
                                end
                                f[ll] = f[ll] + c * f[ii]
                            end
                        end
                        ll = ll + 1
                    end
                    ii = ii + 1
                end
            """
            # do back substitution of upper diagonal A to obtain a
            """
                iii = 0
                for iii_loop
            """
            # jjj starts at the bottom row and works upwards
            """
                    jjj = {nDOFs} - iii - 1  {{id=assign_jjj}}
                    a[jjj] = f[jjj]   {{id=set_a, dep=assign_jjj}}
                    for kkk_loop
                        if kkk_loop > {nDOFs} - iii_loop - 1
                            a[jjj] = a[jjj] - A[jjj,kkk_loop] * a[kkk_loop]
                        end
                    end
                    a[jjj] = a[jjj] / A[jjj,jjj]
                    iii = iii + 1
                end
            end
            """
            # Do final loop to assign output values
            """
            for iiii
            """
            # Having found a, this gives us the coefficients for the Taylor expansion with the actual coordinates.
            """
                if NUM_EXT[0] > 0.0
                {act_coord_expr}
            """
            # if element is not external, just use old field values.
            """
                else
                    DG1[iiii] = DG1_OLD[iiii]
                end
            end
            """).format(**shapes)

        self._kernel = (domain, instrs)

    def apply(self, v_DG1_old, v_DG1, act_coords, eff_coords, num_ext):
        """
        Perform the par loop.

        Takes a scalar field in DG1, the coordinates of the equispaced DG1 DoFs
        and the "effective" coordinates of those DoFs. These "effective"
        coordinates correspond to the locations of the DoFs from the original
        space (i.e. the one being recovered from), with the recovery process
        applied to them.

        Args:
            v_DG1_old (:class:`Function`): the originally recovered field in the
                equispaced DG1 :class:`FunctionSpace`.
            v_DG1 (:class:`Function`): the output field in the equispaced DG1
                :class:`FunctionSpace`.
            act_coords (:class:`Function`): a field whose values contain the
                actual coordinates (in Cartesian components) of the DoFs of the
                equispaced DG1 :class:`FunctionSpace`. This field should be in
                the equispaced DG1 :class:`VectorFunctionSpace`.
            eff_coords (:class:`Function`): a field whose values contain
                coordinates (in Cartesian components) of the DoFs corresponding
                to the effective locations of the original recovery process.
                This field  should be in the equispaced DG1
                :class:`VectorFunctionSpace`.
            num_ext (:class:`Function`): a field in the DG0
                :class:`FunctionSpace` whose DoFs contain the number of DoFs of
                the equispaced DG1 :class:`FunctionSpace` that are on the
                exterior of the domain.
        """
        par_loop(self._kernel, dx,
                 {"DG1_OLD": (v_DG1_old, READ),
                  "DG1": (v_DG1, WRITE),
                  "ACT_COORDS": (act_coords, READ),
                  "EFF_COORDS": (eff_coords, READ),
                  "NUM_EXT": (num_ext, READ)},
                 is_loopy_kernel=True)


class Average(object):
    """
    Evaluates values at DoFs shared between elements using averaging.

    This kernel restores the continuity of a broken field using an averaging
    operation. The values of DoFs shared between by elements are computed as the
    average of the corresponding DoFs of the discontinuous field.
    """
    def __init__(self, V):
        """
        Args:
            V (:class:`FunctionSpace`): The :class:`FunctionSpace` of the target
                field for the :class:`Averager` operator.
        """
        shapes = {"nDOFs": V.finat_element.space_dimension(),
                  "dim": np.prod(V.shape, dtype=int)}

        domain = "{{[i, j]: 0 <= i < {nDOFs} and 0 <= j < {dim}}}".format(**shapes)

        # Loop over node extent and dof extent
        # vo is v_out, v is the function in, w is the weight
        # NOTE: Any bcs on the function v should just work.
        instrs = (
            """
            for i
                for j
                    vo[i,j] = vo[i,j] + v[i,j] / w[i,j]
                end
            end
            """)

        self._kernel = (domain, instrs)

    def apply(self, v_out, weighting, v_in):
        """
        Performs the par_loop.

        Args:
            v_out (:class:`Function`): the continuous output field.
            weighting (:class:`Function`): the weights to be used for the
                averaging.
            v_in (:class:`Function`): the (discontinuous) input field.
        """
        par_loop(self._kernel, dx,
                 {"vo": (v_out, INC),
                  "w": (weighting, READ),
                  "v": (v_in, READ)},
                 is_loopy_kernel=True)


class AverageWeightings(object):
    """
    Finds the weights for the :class:`Averager` operator.

    This computes the weights used in the averaging process for each DoF. This
    is the multiplicity of the DoFs -- as in how many elements each DoF is
    shared between.
    """

    def __init__(self, V):
        """
        Args:
            V (:class:`FunctionSpace`): the continuous function space in which
                the target field of the averaging process lives.
        """
        shapes = {"nDOFs": V.finat_element.space_dimension(),
                  "dim": np.prod(V.shape, dtype=int)}

        domain = "{{[i, j]: 0 <= i < {nDOFs} and 0 <= j < {dim}}}".format(**shapes)

        # w is the weights
        instrs = (
            """
            for i
                for j
                    w[i,j] = w[i,j] + 1.0
                end
            end
            """)

        self._kernel = (domain, instrs)

    def apply(self, w):
        """
        Performs the par loop.

        Args:
            w (:class:`Function`): the field in which to store the weights. This
                lives in the continuous target space.
        """
        par_loop(self._kernel, dx,
                 {"w": (w, INC)},
                 is_loopy_kernel=True)


class PhysicsRecoveryTop():
    """
    Performs the "physics" boundary recovery at the domain's top boundary.

    A kernel for improving the accuracy at the top boundary of the domain for
    the operator for recovering a field from the lowest-order density space to
    the lowest-order temperature space. The kernel is called as part of the
    "physics" method of the :class:`BoundaryRecovery` operator.
    """

    def __init__(self):

        domain = ("{[i]: 0 <= i < 1}")

        # CG1 is the uncorrected field that has been originally recovered
        # DG1 is the corrected output field
        instrs = (
            """
            DG1[0] = CG1[0]
            DG1[1] = -CG1[0] + 2 * CG1[1]
            """)

        self._kernel = (domain, instrs)

    def apply(self, v_DG1, v_CG1):
        """
        Performs the par loop.

        Args:
            v_DG1 (:class:`Function`): the target field to be corrected. It
                should be in a discontinuous :class:`FunctionSpace`.
            v_CG1 (:class:`Function`): the uncorrected field (after the initial
                recovery process). It should be in a continuous
                :class:`FunctionSpace`.
        """
        par_loop(self._kernel, dx,
                 args={"DG1": (v_DG1, WRITE),
                       "CG1": (v_CG1, READ)},
                 is_loopy_kernel=True,
                 iteration_region=ON_TOP)


class PhysicsRecoveryBottom():
    """
    Performs the "physics" boundary recovery at the domain's bottom boundary.

    A kernel for improving the accuracy at the bottom boundary of the domain for
    the operator for recovering a field from the lowest-order density space to
    the lowest-order temperature space. The kernel is called as part of the
    "physics" method of the :class:`BoundaryRecovery` operator.
    """

    def __init__(self):

        domain = ("{[i]: 0 <= i < 1}")

        # CG1 is the uncorrected field that has been originally recovered
        # DG1 is the corrected output field
        instrs = (
            """
            DG1[0] = 2 * CG1[0] - CG1[1]
            DG1[1] = CG1[1]
            """)

        self._kernel = (domain, instrs)

    def apply(self, v_DG1, v_CG1):
        """
        Performs the par loop.

        Args:
            v_DG1 (:class:`Function`): the target field to be corrected. It
                should be in a discontinuous :class:`FunctionSpace`.
            v_CG1 (:class:`Function`): the uncorrected field (after the initial
                recovery process). It should be in a continuous
                :class:`FunctionSpace`.
        """
        par_loop(self._kernel, dx,
                 args={"DG1": (v_DG1, WRITE),
                       "CG1": (v_CG1, READ)},
                 is_loopy_kernel=True,
                 iteration_region=ON_BOTTOM)


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
                :class:`FunctionSpace`space whose vertex values have already
                been limited.
            field_old (:class:`Function`): The original un-limited field in the
                broken temperature :class:`FunctionSpace`.
        """
        par_loop(self._kernel, dx,
                 {"field_hat": (field_hat, WRITE),
                  "field_DG1": (field_DG1, READ),
                  "field_old": (field_old, READ)},
                 is_loopy_kernel=True)
