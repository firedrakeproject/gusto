from firedrake import (BrokenElement, Function, FunctionSpace, interval,
                       FiniteElement, TensorProductElement)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from gusto.kernels import LimitMidpoints

__all__ = ["ThetaLimiter", "NoLimiter"]


class ThetaLimiter(object):
    """
    A vertex based limiter for fields in the DG1xCG2 space, i.e. temperature
    variables in the next-to-lowest order set of spaces. This acts like the
    vertex-based limiter implemented in Firedrake, but in addition corrects
    the central nodes to prevent new maxima or minima forming.
    """

    def __init__(self, space):
        """
        Initialise limiter
        :arg space: the space in which the transported variables lies.
                    It should be a form of the DG1xCG2 space.
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

        # if sub_elements[0].variant() == 'equispaced':
        #     raise ValueError('Theta Limiter can only be used with an equispaced DG1 space in the horizontal')

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
        The application of the limiter to the theta-space field.
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
    """
    A blank limiter that does nothing.
    """

    def __init__(self):
        """
        Initialise the blank limiter.
        """
        pass

    def apply(self, field):
        """
        The application of the blank limiter.
        """
        pass
