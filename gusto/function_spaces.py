"""
This module contains routines to generate the compatible function spaces to be
used by the model.
"""

from firedrake import (HDiv, FunctionSpace, FiniteElement, TensorProductElement,
                       interval)

# TODO: there is danger here for confusion about degree, particularly for the CG
# spaces -- does a "CG" space with degree = 1 mean the "CG" space in the de Rham
# complex of degree 1 ("CG3"), or "CG1"?
# TODO: would it be better to separate creation of specific named spaces from
# the creation of the de Rham complex spaces?
# TODO: how do we create HCurl spaces if we want them?

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

    def __call__(self, name, family=None, horizontal_degree=None,
                 vertical_degree=None, V=None):
        """
        Returns a space, and also creates it if it is not created yet.

        If a space needs creating, it may be that more arguments (such as the
        family and degree) need to be provided. Alternatively a space can be
        passed in to be stored in the creator.

        Args:
            name (str): the name of the space.
            family (str, optional): name of the finite element family to be
                created. Defaults to None.
            horizontal_degree (int, optional): the horizontal degree of the
                finite element space to be created. Defaults to None.
            vertical_degree (int, optional): the vertical degree of the
                finite element space to be created. Defaults to None.
            V (:class:`FunctionSpace`, optional): an existing space, to be
                stored in the creator object. If this is provided, it will be
                added to the creator and no other action will be taken. This
                space will be returned. Defaults to None.

        Returns:
            :class:`FunctionSpace`: the desired function space.
        """

        try:
            # First attempt to return the space based on the name, if it exists
            return getattr(self, name)

        except AttributeError:

            # Space does not exist in creator
            if V is not None:
                # The space itself has been provided (to add it to the creator)
                value = V

            elif name == "DG1_equispaced":
                # Special case based on name
                if self.extruded_mesh:
                    value = self.build_dg_space(1, 1, variant='equispaced')
                else:
                    value = self.build_dg_space(1, variant='equispaced')

            else:
                # Need to create space, based on name/family/degree
                assert horizontal_degree is not None

                # Loop through name and family combinations
                if name == "HDiv" and family in ["BDM", "RT", "CG", "RTCF"]:
                    value = self.build_hdiv_space(family, horizontal_degree, vertical_degree)
                elif name == "theta":
                    value = self.build_theta_space(horizontal_degree, vertical_degree)
                elif family == "DG":
                    value = self.build_dg_space(horizontal_degree, vertical_degree)
                elif family == "CG":
                    value = self.build_cg_space(horizontal_degree, vertical_degree)
                else:
                    raise ValueError(f'State has no space corresponding to {name}')
            setattr(self, name, value)
            return value

    def build_compatible_spaces(self, family, horizontal_degree,
                                vertical_degree=None):
        """
        Builds the sequence of compatible finite element spaces for the mesh.

        If the mesh is not extruded, this builds and returns the spaces:
            (HDiv, DG).
        If the mesh is extruded, this builds and returns the following spaces:
            (HDiv, DG, theta).
        The 'theta' space corresponds to the vertical component of the velocity.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the DG space.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the DG space. Defaults to None. Must be
                specified if the mesh is extruded.

        Returns:
            tuple: the created compatible :class:`FunctionSpace` objects.
        """
        if self.extruded_mesh and not self._initialised_base_spaces:
            self.build_base_spaces(family, horizontal_degree, vertical_degree)
            Vu = self.build_hdiv_space(family, horizontal_degree, vertical_degree)
            setattr(self, "HDiv", Vu)
            Vdg = self.build_dg_space(horizontal_degree, vertical_degree)
            setattr(self, "DG", Vdg)
            Vth = self.build_theta_space(horizontal_degree, vertical_degree)
            setattr(self, "theta", Vth)
            return Vu, Vdg, Vth
        else:
            Vu = self.build_hdiv_space(family, horizontal_degree+1)
            setattr(self, "HDiv", Vu)
            Vdg = self.build_dg_space(horizontal_degree, vertical_degree)
            setattr(self, "DG", Vdg)
            return Vu, Vdg

    def build_base_spaces(self, family, horizontal_degree, vertical_degree):
        """
        Builds the :class:`FiniteElement` objects for the base mesh.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the DG space.
            vertical_degree (int): the polynomial degree of the vertical part of
                the DG space.
        """
        cell = self.mesh._base_mesh.ufl_cell().cellname()

        # horizontal base spaces
        self.S1 = FiniteElement(family, cell, horizontal_degree+1)
        self.S2 = FiniteElement("DG", cell, horizontal_degree)

        # vertical base spaces
        self.T0 = FiniteElement("CG", interval, vertical_degree+1)
        self.T1 = FiniteElement("DG", interval, vertical_degree)

        self._initialised_base_spaces = True

    def build_hdiv_space(self, family, horizontal_degree, vertical_degree=None):
        """
        Builds and returns the HDiv :class:`FunctionSpace`.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the DG space from the de Rham complex.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the the DG space from the de Rham complex.
                Defaults to None. Must be specified if the mesh is extruded.

        Returns:
            :class:`FunctionSpace`: the HDiv space.
        """
        if self.extruded_mesh:
            if not self._initialised_base_spaces:
                if vertical_degree is None:
                    raise ValueError('vertical_degree must be specified to create HDiv space on an extruded mesh')
                self.build_base_spaces(family, horizontal_degree, vertical_degree)
            Vh_elt = HDiv(TensorProductElement(self.S1, self.T1))
            Vt_elt = TensorProductElement(self.S2, self.T0)
            Vv_elt = HDiv(Vt_elt)
            V_elt = Vh_elt + Vv_elt
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement(family, cell, horizontal_degree)
        return FunctionSpace(self.mesh, V_elt, name='HDiv')

    def build_dg_space(self, horizontal_degree, vertical_degree=None, variant=None):
        """
        Builds and returns the DG :class:`FunctionSpace`.

        Args:
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the DG space.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the DG space. Defaults to None. Must be
                specified if the mesh is extruded.
            variant (str): the variant of the underlying :class:`FiniteElement`
                to use. Defaults to None, which will call the default variant.

        Returns:
            :class:`FunctionSpace`: the DG space.
        """
        if self.extruded_mesh:
            if vertical_degree is None:
                raise ValueError('vertical_degree must be specified to create DG space on an extruded mesh')
            if not self._initialised_base_spaces or self.T1.degree() != vertical_degree or self.T1.variant() != variant:
                cell = self.mesh._base_mesh.ufl_cell().cellname()
                S2 = FiniteElement("DG", cell, horizontal_degree, variant=variant)
                T1 = FiniteElement("DG", interval, vertical_degree, variant=variant)
            else:
                S2 = self.S2
                T1 = self.T1
            V_elt = TensorProductElement(S2, T1)
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement("DG", cell, horizontal_degree, variant=variant)
        # TODO: how should we name this if vertical degree is different?
        name = f'DG{horizontal_degree}_equispaced' if variant == 'equispaced' else f'DG{horizontal_degree}'
        return FunctionSpace(self.mesh, V_elt, name=name)

    def build_theta_space(self, horizontal_degree, vertical_degree):
        """
        Builds and returns the 'theta' space.

        This corresponds to the non-Piola mapped space of the vertical component
        of the velocity. The space will be discontinuous in the horizontal but
        continuous in the vertical.

        Args:
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the DG space from the de Rham complex.
            vertical_degree (int): the polynomial degree of the vertical part of
                the DG space from the de Rham complex.

        Raises:
            AssertionError: the mesh is not extruded.

        Returns:
            :class:`FunctionSpace`: the 'theta' space.
        """
        assert self.extruded_mesh, 'Cannot create theta space if mesh is not extruded'
        if not self._initialised_base_spaces:
            cell = self.mesh._base_mesh.ufl_cell().cellname()
            self.S2 = FiniteElement("DG", cell, horizontal_degree)
            self.T0 = FiniteElement("CG", interval, vertical_degree+1)
        V_elt = TensorProductElement(self.S2, self.T0)
        return FunctionSpace(self.mesh, V_elt, name='Vtheta')

    def build_cg_space(self, horizontal_degree, vertical_degree):
        """
        Builds the continuous scalar space at the top of the de Rham complex.

        Args:
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the DG space from the de Rham complex.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the the DG space from the de Rham complex.
                Defaults to None. Must be specified if the mesh is extruded.

        Returns:
            :class:`FunctionSpace`: the continuous space.
        """

        if self.extruded_mesh:
            if vertical_degree is None:
                raise ValueError('vertical_degree must be specified to create CG space on an extruded mesh')
            cell = self.mesh._base_mesh.ufl_cell().cellname()
            CG_hori = FiniteElement("CG", cell, horizontal_degree+1)
            CG_vert = FiniteElement("CG", interval, vertical_degree+1)
            V_elt = TensorProductElement(CG_hori, CG_vert)
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement("DG", cell, horizontal_degree+1, variant=variant)

        # How should we name this if the horizontal and vertical degrees are different?
        name = f'CG{horizontal_degree+1}'

        return FunctionSpace(self.mesh, V_elt, name=name)
