"""
This module contains routines to generate the compatible function spaces to be
used by the model.
"""

from gusto import logger
from firedrake import (HCurl, HDiv, FunctionSpace, FiniteElement,
                       TensorProductElement, interval)

__all__ = ["Spaces", "check_degree_args"]

# HDiv spaces are keys, HCurl spaces are values
hdiv_hcurl_dict = {'RT': 'RTE',
                   'RTE': 'RTE',
                   'RTF': 'RTE',
                   'BDM': 'BDME',
                   'BDME': 'BDME',
                   'BDMF': 'BDME',
                   'RTCF': 'RTCE',
                   'RTCE': 'RTCE',
                   'CG': 'DG',
                   'BDFM': None}

# HCurl spaces are keys, HDiv spaces are values
# Can't just reverse the other dictionary as values are not necessarily unique
hcurl_hdiv_dict = {'RT': 'RTF',
                   'RTE': 'RTF',
                   'RTF': 'RTF',
                   'BDM': 'BDMF',
                   'BDME': 'BDMF',
                   'BDMF': 'BDMF',
                   'RTCE': 'RTCF',
                   'RTCF': 'RTCF',
                   'CG': 'CG',
                   'BDFM': 'BDFM'}


# Degree to use for H1 space for a particular family
def h1_degree(family, l2_degree):
    """
    Return the degree of the H1 space, for a particular de Rham complex.

    Args:
        family (str): the family of the HDiv or HCurl elements.
        l2_degree (int): the degree of the L2 space at the bottom of the complex

    Returns:
        int: the degree of the H1 space at the top of the complex.
    """
    if family in ['CG', 'RT', 'RTE', 'RTF', 'RTCE', 'RTCF']:
        return l2_degree + 1
    elif family in ['BDM', 'BDME', 'BDMF']:
        return l2_degree + 2
    elif family == 'BDFM':
        return l2_degree + 1
    else:
        raise ValueError(f'family {family} not recognised')


class Spaces(object):
    """Object to create and hold the model's finite element spaces."""
    def __init__(self, mesh):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
        """
        self.mesh = mesh
        self.extruded_mesh = hasattr(mesh, "_base_mesh")

        # Make dictionaries to store base elements for extruded meshes
        # Here the keys are the horizontal and vertical degrees
        self._initialised_base_spaces = {}
        self.base_elt_hori_hdiv = {}
        self.base_elt_hori_hcurl = {}
        self.base_elt_hori_dg = {}
        self.base_elt_hori_cg = {}
        self.base_elt_vert_cg = {}
        self.base_elt_vert_dg = {}

    def __call__(self, name, family=None, degree=None,
                 horizontal_degree=None, vertical_degree=None,
                 V=None, overwrite_space=False):
        """
        Returns a space, and also creates it if it is not created yet.

        If a space needs creating, it may be that more arguments (such as the
        family and degree) need to be provided. Alternatively a space can be
        passed in to be stored in the space container.

        For extruded meshes, it is possible to seperately specify the horizontal
        and vertical degrees of the elements. Alternatively, if these degrees
        should be the same then this can be specified through the "degree"
        argument.

        Args:
            name (str): the name of the space.
            family (str, optional): name of the finite element family to be
                created. Defaults to None.
            degree (int, optional): the element degree used for the space.
                Defaults to None, in which case the horizontal degree must be
                provided.
            horizontal_degree (int, optional): the horizontal degree of the
                finite element space to be created. Defaults to None.
            vertical_degree (int, optional): the vertical degree of the
                finite element space to be created. Defaults to None.
            V (:class:`FunctionSpace`, optional): an existing space, to be
                stored in the creator object. If this is provided, it will be
                added to the creator and no other action will be taken. This
                space will be returned. Defaults to None.
            overwrite_space (bool, optional): Logical to allow space existing in
                container to be overwritten by an incoming space. Defaults to
                False.

        Returns:
            :class:`FunctionSpace`: the desired function space.
        """

        implemented_families = ["DG", "CG", "RT", "RTF", "RTE", "RTCF", "RTCE",
                                "BDM", "BDMF", "BDME", "BDFM"]
        if family not in [None]+implemented_families:
            raise NotImplementedError(f'family {family} either not recognised '
                                      + 'or implemented in Gusto')

        if hasattr(self, name) and (V is None or not overwrite_space):
            # We have requested a space that should already have been created
            if V is not None:
                assert getattr(self, name) == V, \
                    f'There is a conflict between the space {name} already ' + \
                    'existing in the space container, and the space being passed to it'
            return getattr(self, name)

        else:
            # Space does not exist in creator
            if V is not None:
                # The space itself has been provided (to add it to the creator)
                value = V

            elif name == "DG1_equispaced":
                # Special case as no degree arguments need providing
                value = self.build_l2_space(1, 1, variant='equispaced', name='DG1_equispaced')

            else:
                check_degree_args('Spaces', self.mesh, degree, horizontal_degree, vertical_degree)

                # Convert to horizontal and vertical degrees
                horizontal_degree = degree if horizontal_degree is None else horizontal_degree
                vertical_degree = degree if vertical_degree is None else vertical_degree

                # Loop through name and family combinations
                if name in ["HDiv", "HCurl", "theta", "L2", "H1"]:
                    # This is one of the compatible finite element spaces, so
                    # look for the space in the de Rham dictionary
                    degree_pair = (horizontal_degree, vertical_degree)
                    # If the de Rham complex doesn't exist yet, create it
                    if degree_pair not in self._de_rham_dict.keys():
                        self._de_rham_dict[degree_pair] = \
                            DeRhamComplex(self.mesh, hdiv_family,
                                          horizontal_degree, vertical_degree)

                    de_rham_complex = self._de_rham_dict[degree_pair]
                    value = de_rham_complex(name)

                if name == "HDiv" and family in ["BDM", "RT", "CG", "RTCF", "RTF", "BDMF"]:
                    hdiv_family = hcurl_hdiv_dict[family]
                    value = self.build_hdiv_space(hdiv_family, horizontal_degree, vertical_degree)
                elif name == "HCurl" and family in ["BDM", "RT", "CG", "RTCE", "RTE", "BDME"]:
                    hcurl_family = hdiv_hcurl_dict[family]
                    value = self.build_hcurl_space(hcurl_family, horizontal_degree, vertical_degree)
                elif name == "theta":
                    value = self.build_theta_space(horizontal_degree, vertical_degree)
                elif family == "DG":
                    value = self.build_l2_space(horizontal_degree, vertical_degree, name=name)
                elif family == "CG":
                    value = self.build_h1_space(family, horizontal_degree, vertical_degree, name=name)
                else:
                    raise ValueError(f'There is no space corresponding to {name}')
            setattr(self, name, value)
            return value

    def build_compatible_spaces(self, family, horizontal_degree,
                                vertical_degree=None):
        """
        Builds the sequence of compatible finite element spaces for the mesh.

        If the mesh is not extruded, this builds and returns the spaces:      \n
        (H1, HCurl, HDiv, L2).                                                \n
        If the mesh is extruded, this builds and returns the spaces:          \n
        (H1, HCurl, HDiv, L2, theta).                                         \n
        The 'theta' space corresponds to the vertical component of the velocity.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the L2 space.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the L2 space. Defaults to None. Must be
                specified if the mesh is extruded.

        Returns:
            tuple: the created compatible :class:`FunctionSpace` objects.
        """
        if (horizontal_degree, vertical_degree) in self._initialised_base_spaces.keys():
            pass

        elif self.extruded_mesh:
            # Base spaces need building, while horizontal and vertical degrees
            # need specifying separately. Vtheta needs returning.
            self.build_base_spaces(family, horizontal_degree, vertical_degree)
            Vcg = self.build_h1_space(family,
                                      h1_degree(family, horizontal_degree),
                                      vertical_degree+1, name='H1')
            setattr(self, "H1", Vcg)
            hcurl_family = hdiv_hcurl_dict[family]
            Vcurl = self.build_hcurl_space(hcurl_family, horizontal_degree, vertical_degree)
            setattr(self, "HCurl", Vcurl)
            hdiv_family = hcurl_hdiv_dict[family]
            Vu = self.build_hdiv_space(hdiv_family, horizontal_degree, vertical_degree)
            setattr(self, "HDiv", Vu)
            Vdg = self.build_l2_space(horizontal_degree, vertical_degree, name='L2')
            setattr(self, "L2", Vdg)
            setattr(self, "DG", Vdg)  # Register this as "L2" and "DG"
            Vth = self.build_theta_space(horizontal_degree, vertical_degree)
            setattr(self, "theta", Vth)

            return Vcg, Vcurl, Vu, Vdg, Vth

        elif self.mesh.topological_dimension() > 1:
            # 2D: two de Rham complexes (hcurl or hdiv) with 3 spaces
            # 3D: one de Rham complexes with 4 spaces
            # either way, build all spaces
            Vcg = self.build_h1_space(family, h1_degree(family, horizontal_degree), name='H1')
            setattr(self, "H1", Vcg)
            hcurl_family = hdiv_hcurl_dict[family]
            Vcurl = self.build_hcurl_space(hcurl_family, horizontal_degree)
            setattr(self, "HCurl", Vcurl)
            hdiv_family = hcurl_hdiv_dict[family]
            Vu = self.build_hdiv_space(family, horizontal_degree)
            setattr(self, "HDiv", Vu)
            Vdg = self.build_l2_space(horizontal_degree, vertical_degree, name='L2')
            setattr(self, "L2", Vdg)
            setattr(self, "DG", Vdg)  # Register this as "L2" and "DG"

            return Vcg, Vcurl, Vu, Vdg

        else:
            # 1D domain, de Rham complex has 2 spaces
            # CG, hdiv and hcurl spaces should be the same
            Vcg = self.build_h1_space(family, horizontal_degree+1, name='H1')
            setattr(self, "H1", Vcg)
            setattr(self, "HCurl", None)
            setattr(self, "HDiv", Vcg)
            Vdg = self.build_l2_space(horizontal_degree, name='L2')
            setattr(self, "L2", Vdg)
            setattr(self, "DG", Vdg)  # Register this as "L2" and "DG"

            return Vcg, Vdg

    def build_base_spaces(self, family, horizontal_degree, vertical_degree):
        """
        Builds the :class:`FiniteElement` objects for the base mesh.

        Args:
            family (str): the family of the horizontal part of either the HDiv
                or HCurl space.
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the L2 space.
            vertical_degree (int): the polynomial degree of the vertical part of
                the L2 space.
        """

        if family == 'BDFM':
            # Need a special implementation of base spaces here as it does not
            # fit the same pattern as other spaces
            self.build_bdfm_base_spaces(horizontal_degree, vertical_degree)
            return

        cell = self.mesh._base_mesh.ufl_cell().cellname()

        hdiv_family = hcurl_hdiv_dict[family]
        hcurl_family = hdiv_hcurl_dict[family]

        # horizontal base spaces
        if horizontal_degree not in self.base_elt_hori_dg.keys():
            key = horizontal_degree
            self.base_elt_hori_hdiv[key] = FiniteElement(hdiv_family, cell, horizontal_degree+1)
            self.base_elt_hori_hcurl[key] = FiniteElement(hcurl_family, cell, horizontal_degree+1)
            self.base_elt_hori_dg[key] = FiniteElement("DG", cell, horizontal_degree)
            self.base_elt_hori_cg[key] = FiniteElement("CG", cell, h1_degree(family, horizontal_degree))

        # vertical base spaces
        if vertical_degree not in self.base_elt_vert_dg.keys():
            key = vertical_degree
            self.base_elt_vert_cg[key] = FiniteElement("CG", interval, vertical_degree+1)
            self.base_elt_vert_dg[key] = FiniteElement("DG", interval, vertical_degree)

        degree_pair = (horizontal_degree, vertical_degree)
        self._initialised_base_spaces[degree_pair] = True

    def build_hcurl_space(self, family, horizontal_degree, vertical_degree=None):
        """
        Builds and returns the HCurl :class:`FunctionSpace`.

        Args:
            family (str): the family of the horizontal part of the HCurl space.
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the L2 space from the de Rham complex.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the L2 space from the de Rham complex.
                Defaults to None. Must be specified if the mesh is extruded.

        Returns:
            :class:`FunctionSpace`: the HCurl space.
        """
        if family is None:
            logger.warning('There is no HCurl space for this family. Not creating one')
            return None

        if self.extruded_mesh:
            if (horizontal_degree, vertical_degree) not in self._initialised_base_spaces.keys():
                if vertical_degree is None:
                    raise ValueError('vertical_degree must be specified to create HCurl space on an extruded mesh')
                self.build_base_spaces(family, horizontal_degree, vertical_degree)

            Vh_elt = HCurl(TensorProductElement(self.base_elt_hori_hcurl[horizontal_degree],
                                                self.base_elt_vert_cg[vertical_degree]))
            Vv_elt = HCurl(TensorProductElement(self.base_elt_hori_cg[horizontal_degree],
                                                self.base_elt_vert_dg[vertical_degree]))
            V_elt = Vh_elt + Vv_elt
        else:
            cell = self.mesh.ufl_cell().cellname()
            hcurl_family = hdiv_hcurl_dict[family]
            V_elt = FiniteElement(hcurl_family, cell, horizontal_degree)

        return FunctionSpace(self.mesh, V_elt, name='HCurl')

    def build_hdiv_space(self, family, horizontal_degree, vertical_degree=None):
        """
        Builds and returns the HDiv :class:`FunctionSpace`.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the L2 space from the de Rham complex.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the L2 space from the de Rham complex.
                Defaults to None. Must be specified if the mesh is extruded.

        Returns:
            :class:`FunctionSpace`: the HDiv space.
        """
        if self.extruded_mesh:
            if (horizontal_degree, vertical_degree) not in self._initialised_base_spaces.keys():
                if vertical_degree is None:
                    raise ValueError('vertical_degree must be specified to create HDiv space on an extruded mesh')
                self.build_base_spaces(family, horizontal_degree, vertical_degree)
            Vh_elt = HDiv(TensorProductElement(self.base_elt_hori_hdiv[horizontal_degree],
                                               self.base_elt_vert_dg[vertical_degree]))
            Vt_elt = TensorProductElement(self.base_elt_hori_dg[horizontal_degree],
                                          self.base_elt_vert_cg[vertical_degree])
            Vv_elt = HDiv(Vt_elt)
            V_elt = Vh_elt + Vv_elt
        else:
            cell = self.mesh.ufl_cell().cellname()
            hdiv_family = hcurl_hdiv_dict[family]
            V_elt = FiniteElement(hdiv_family, cell, horizontal_degree+1)
        return FunctionSpace(self.mesh, V_elt, name='HDiv')

    def build_l2_space(self, horizontal_degree, vertical_degree=None, variant=None, name='L2'):
        """
        Builds and returns the discontinuous L2 :class:`FunctionSpace`.

        Args:
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the L2 space.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the L2 space. Defaults to None. Must be
                specified if the mesh is extruded.
            variant (str, optional): the variant of the underlying
                :class:`FiniteElement` to use. Defaults to None, which will call
                the default variant.
            name (str, optional): name to assign to the function space. Default
                is "L2".

        Returns:
            :class:`FunctionSpace`: the L2 space.
        """
        assert not hasattr(self, name), f'There already exists a function space with name {name}'

        if self.extruded_mesh:
            if vertical_degree is None:
                raise ValueError('vertical_degree must be specified to create L2 space on an extruded mesh')
            if ((horizontal_degree, vertical_degree) not in self._initialised_base_spaces.keys()
                    or self.base_elt_vert_dg[vertical_degree].variant() != variant
                    or self.base_elt_hori_dg[horizontal_degree].degree() != variant):
                cell = self.mesh._base_mesh.ufl_cell().cellname()
                base_elt_hori_dg = FiniteElement("DG", cell, horizontal_degree, variant=variant)
                base_elt_vert_dg = FiniteElement("DG", interval, vertical_degree, variant=variant)
            else:
                base_elt_hori_dg = self.base_elt_hori_dg[horizontal_degree]
                base_elt_vert_dg = self.base_elt_vert_dg[vertical_degree]
            V_elt = TensorProductElement(base_elt_hori_dg, base_elt_vert_dg)
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement("DG", cell, horizontal_degree, variant=variant)

        return FunctionSpace(self.mesh, V_elt, name=name)

    def build_theta_space(self, horizontal_degree, vertical_degree):
        """
        Builds and returns the 'theta' space.

        This corresponds to the non-Piola mapped space of the vertical component
        of the velocity. The space will be discontinuous in the horizontal but
        continuous in the vertical.

        Args:
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the L2 space from the de Rham complex.
            vertical_degree (int): the polynomial degree of the vertical part of
                the L2 space from the de Rham complex.

        Raises:
            AssertionError: the mesh is not extruded.

        Returns:
            :class:`FunctionSpace`: the 'theta' space.
        """
        assert self.extruded_mesh, 'Cannot create theta space if mesh is not extruded'
        if (horizontal_degree, vertical_degree) not in self._initialised_base_spaces.keys():
            cell = self.mesh._base_mesh.ufl_cell().cellname()
            base_elt_hori_dg = FiniteElement("DG", cell, horizontal_degree)
            base_elt_vert_cg = FiniteElement("CG", interval, vertical_degree+1)
        else:
            base_elt_hori_dg = self.base_elt_hori_dg[horizontal_degree]
            base_elt_vert_cg = self.base_elt_vert_cg[vertical_degree]
        V_elt = TensorProductElement(base_elt_hori_dg, base_elt_vert_cg)
        return FunctionSpace(self.mesh, V_elt, name='theta')

    def build_h1_space(self, family, horizontal_degree, vertical_degree=None, name='H1'):
        """
        Builds the continuous scalar space at the top of the de Rham complex.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the H1 space.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the H1 space. Defaults to None. Must be
                specified if the mesh is extruded.
            name (str, optional): name to assign to the function space. Default
                is "H1".

        Returns:
            :class:`FunctionSpace`: the continuous space.
        """
        assert not hasattr(self, name), f'There already exists a function space with name {name}'

        if self.extruded_mesh:
            if vertical_degree is None:
                raise ValueError('vertical_degree must be specified to create H1 space on an extruded mesh')
            if (horizontal_degree, vertical_degree) not in self._initialised_base_spaces.keys():
                cell = self.mesh._base_mesh.ufl_cell().cellname()
                base_elt_hori_cg = FiniteElement("CG", cell, horizontal_degree)
                base_elt_vert_cg = FiniteElement("CG", interval, vertical_degree)
            else:
                base_elt_hori_cg = self.base_elt_hori_cg[horizontal_degree]
                base_elt_vert_cg = self.base_elt_vert_cg[vertical_degree]
            V_elt = TensorProductElement(base_elt_hori_cg, base_elt_vert_cg)

        elif family == 'BDFM':
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement("CG", cell, horizontal_degree)
            V_elt += FiniteElement("Bubble", cell, horizontal_degree+1)

        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement("CG", cell, horizontal_degree)

        return FunctionSpace(self.mesh, V_elt, name=name)

    def build_bdfm_base_spaces(self, horizontal_degree, vertical_degree=None):
        """
        Builds the :class:`FiniteElement` objects for the base mesh when using
        the .

        Args:
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the L2 space.
            vertical_degree (int): the polynomial degree of the vertical part of
                the L2 space.
        """

        cell = self.mesh._base_mesh.ufl_cell().cellname()

        hdiv_family = 'BDFM'

        # horizontal base spaces
        key = horizontal_degree
        self.base_elt_hori_hdiv[key] = FiniteElement(hdiv_family, cell, horizontal_degree+1)
        self.base_elt_hori_dg[key] = FiniteElement("DG", cell, horizontal_degree)

        # Add bubble space
        self.base_elt_hori_cg[key] = FiniteElement("CG", cell, horizontal_degree+1)
        self.base_elt_hori_cg[key] += FiniteElement("Bubble", cell, horizontal_degree+2)

        # vertical base spaces
        key = vertical_degree
        self.base_elt_vert_cg[key] = FiniteElement("CG", interval, vertical_degree+1)
        self.base_elt_vert_dg[key] = FiniteElement("DG", interval, vertical_degree)

        self._initialised_base_spaces[(horizontal_degree, vertical_degree)] = True


class DeRhamComplex(object):

    def __init__(self, mesh, horizontal_degree, vertical_degree=None):



def check_degree_args(name, mesh, degree, horizontal_degree, vertical_degree):
    """
    Check the degree arguments passed to either the :class:`Domain` or the
    :class:`Spaces` object. This will raise errors if the arguments are not
    appropriate.

    Args:
        name (str): name of object to print out.
        mesh (:class:`Mesh`): the model's mesh.
        degree (int): the element degree.
        horizontal_degree (int): the element degree used for the horizontal part
            of a space.
        vertical_degree (int): the element degree used for the vertical part
            of a space.
    """

    extruded_mesh = hasattr(mesh, "_base_mesh")

    # Checks on degree arguments
    if degree is None and horizontal_degree is None:
        raise ValueError(f'Either "degree" or "horizontal_degree" must be passed to {name}')
    if extruded_mesh and degree is None and vertical_degree is None:
        raise ValueError(f'For extruded meshes, either "degree" or "vertical_degree" must be passed to {name}')
    if degree is not None and horizontal_degree is not None:
        raise ValueError(f'Cannot pass both "degree" and "horizontal_degree" to {name}')
    if extruded_mesh and degree is not None and vertical_degree is not None:
        raise ValueError(f'Cannot pass both "degree" and "vertical_degree" to {name}')
    if not extruded_mesh and vertical_degree is not None:
        raise ValueError(f'Cannot pass "vertical_degree" to {name} if mesh is not extruded')
