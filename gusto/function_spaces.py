"""
This module contains routines to generate the compatible function spaces to be
used by the model.
"""

from firedrake import (HDiv, FunctionSpace, FiniteElement, TensorProductElement,
                       interval)

class Spaces(object):
    """Object to create and hold the model's finite element spaces."""
    def __init__(self, mesh):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
        """
        self.mesh = mesh
        self.extruded_mesh = hasattr(mesh, "_base_mesh")
        self._initialised_base_spaces = False

    def __call__(self, name, family=None, degree=None, V=None):
        """
        Returns a space, and also creates it if it is not created yet.

        If a space needs creating, it may be that more arguments (such as the
        family and degree) need to be provided. Alternatively a space can be
        passed in to be stored in the creator.

        Args:
            name (str): the name of the space.
            family (str, optional): name of the finite element family to be
                created. Defaults to None.
            degree (int, optional): the degree of the finite element space to be
                created. Defaults to None.
            V (:class:`FunctionSpace`, optional): an existing space, to be
                stored in the creator object. If this is provided, it will be
                added to the creator and no other action will be taken. This
                space will be returned. Defaults to None.

        Returns:
            :class:`FunctionSpace`: the desired function space.
        """

        try:
            return getattr(self, name)
        except AttributeError:
            if V is not None:
                value = V
            elif name == "HDiv" and family in ["BDM", "RT", "CG", "RTCF"]:
                value = self.build_hdiv_space(family, degree)
            elif name == "theta":
                value = self.build_theta_space(degree)
            elif name == "DG1_equispaced":
                value = self.build_dg_space(1, variant='equispaced')
            elif family == "DG":
                value = self.build_dg_space(degree)
            elif family == "CG":
                value = self.build_cg_space(degree)
            else:
                raise ValueError(f'State has no space corresponding to {name}')
            setattr(self, name, value)
            return value

    def build_compatible_spaces(self, family, degree):
        """
        Builds the sequence of compatible finite element spaces for the mesh.

        If the mesh is not extruded, this builds and returns the spaces:
            (HDiv, DG).
        If the mesh is extruded, this builds and returns the following spaces:
            (HDiv, DG, theta).
        The 'theta' space corresponds to the vertical component of the velocity.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            degree (int): the polynomial degree of the DG space.

        Returns:
            tuple: the created compatible :class:`FunctionSpace` objects.
        """
        if self.extruded_mesh and not self._initialised_base_spaces:
            self.build_base_spaces(family, degree)
            Vu = self.build_hdiv_space(family, degree)
            setattr(self, "HDiv", Vu)
            Vdg = self.build_dg_space(degree)
            setattr(self, "DG", Vdg)
            Vth = self.build_theta_space(degree)
            setattr(self, "theta", Vth)
            return Vu, Vdg, Vth
        else:
            Vu = self.build_hdiv_space(family, degree)
            setattr(self, "HDiv", Vu)
            Vdg = self.build_dg_space(degree)
            setattr(self, "DG", Vdg)
            return Vu, Vdg

    def build_base_spaces(self, family, degree):
        """
        Builds the :class:`FiniteElement` objects for the base mesh.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            degree (int): the polynomial degree of the DG space.
        """
        cell = self.mesh._base_mesh.ufl_cell().cellname()

        # horizontal base spaces
        self.S1 = FiniteElement(family, cell, degree+1)
        self.S2 = FiniteElement("DG", cell, degree)

        # vertical base spaces
        self.T0 = FiniteElement("CG", interval, degree+1)
        self.T1 = FiniteElement("DG", interval, degree)

        self._initialised_base_spaces = True

    def build_hdiv_space(self, family, degree):
        """
        Builds and returns the HDiv :class:`FunctionSpace`.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            degree (int): the polynomial degree of the space.

        Returns:
            :class:`FunctionSpace`: the HDiv space.
        """
        if self.extruded_mesh:
            if not self._initialised_base_spaces:
                self.build_base_spaces(family, degree)
            Vh_elt = HDiv(TensorProductElement(self.S1, self.T1))
            Vt_elt = TensorProductElement(self.S2, self.T0)
            Vv_elt = HDiv(Vt_elt)
            V_elt = Vh_elt + Vv_elt
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement(family, cell, degree+1)
        return FunctionSpace(self.mesh, V_elt, name='HDiv')

    def build_dg_space(self, degree, variant=None):
        """
        Builds and returns the DG :class:`FunctionSpace`.

        Args:
            degree (int): the polynomial degree of the space.
            variant (str): the variant of the underlying :class:`FiniteElement`
                to use. Defaults to None, which will call the default variant.

        Returns:
            :class:`FunctionSpace`: the DG space.
        """
        if self.extruded_mesh:
            if not self._initialised_base_spaces or self.T1.degree() != degree or self.T1.variant() != variant:
                cell = self.mesh._base_mesh.ufl_cell().cellname()
                S2 = FiniteElement("DG", cell, degree, variant=variant)
                T1 = FiniteElement("DG", interval, degree, variant=variant)
            else:
                S2 = self.S2
                T1 = self.T1
            V_elt = TensorProductElement(S2, T1)
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement("DG", cell, degree, variant=variant)
        name = f'DG{degree}_equispaced' if variant == 'equispaced' else f'DG{degree}'
        return FunctionSpace(self.mesh, V_elt, name=name)

    def build_theta_space(self, degree):
        """
        Builds and returns the 'theta' space.

        This corresponds to the non-Piola mapped space of the vertical component
        of the velocity. The space will be discontinuous in the horizontal but
        continuous in the vertical.

        Args:
            degree (int): degree of the corresponding density space.

        Raises:
            AssertionError: the mesh is not extruded.

        Returns:
            :class:`FunctionSpace`: the 'theta' space.
        """
        assert self.extruded_mesh
        if not self._initialised_base_spaces:
            cell = self.mesh._base_mesh.ufl_cell().cellname()
            self.S2 = FiniteElement("DG", cell, degree)
            self.T0 = FiniteElement("CG", interval, degree+1)
        V_elt = TensorProductElement(self.S2, self.T0)
        return FunctionSpace(self.mesh, V_elt, name='Vtheta')

    def build_cg_space(self, degree):
        """
        Builds the continuous scalar space at the top of the de Rham complex.

        Args:
            degree (int): degree of the continuous space.

        Returns:
            :class:`FunctionSpace`: the continuous space.
        """
        return FunctionSpace(self.mesh, "CG", degree, name=f'CG{degree}')
