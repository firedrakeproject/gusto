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
                   'BDFM': None,
                   'BDMCF': 'BDMCE',
                   'BDMCE': 'BDMCE'}

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
                   'BDFM': 'BDFM',
                   'BDMCF': 'BDMCF',
                   'BDMCE': 'BDMCF'}


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
    elif family in ['BDM', 'BDME', 'BDMF', 'BDMCE', 'BDMCF']:
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
        self.de_rham_complex = {}

    def __call__(self, name):
        """
        Returns a space, and also creates it if it is not created yet.

        Args:
            name (str): the name of the space.

        Returns:
            :class:`FunctionSpace`: the desired function space.
        """

        if hasattr(self, name):
            return getattr(self, name)

        else:
            raise ValueError(f'The space container has no space {name}')

    def add_space(self, name, space, overwrite_space=False):
        """
        Adds a function space to the container.

        Args:
            name (str): the name of the space.
            space (:class:`FunctionSpace`): the function space to be added to
                the Space container..
            overwrite_space (bool, optional): Logical to allow space existing in
                container to be overwritten by an incoming space. Defaults to
                False.
        """

        if hasattr(self, name) and not overwrite_space:
            raise RuntimeError(f'Space {name} already exists. If you really '
                               + 'to create it then set `overwrite_space` as '
                               + 'to be True')

        setattr(self, name, space)

    def create_space(self, name, family, degree, overwrite_space=False):
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
            family (str): name of the finite element family to be created.
            degree (int): the element degree used for the space.
            overwrite_space (bool, optional): Logical to allow space existing in
                container to be overwritten by an incoming space. Defaults to
                False.

        Returns:
            :class:`FunctionSpace`: the desired function space.
        """

        if hasattr(self, name) and not overwrite_space:
            raise RuntimeError(f'Space {name} already exists. If you really '
                               + 'to create it then set `overwrite_space` as '
                               + 'to be True')

        space = FunctionSpace(self.mesh, family, degree, name=name)
        setattr(self, name, space)
        return space

    def build_compatible_spaces(self, family, horizontal_degree,
                                vertical_degree=None, complex_name=None):
        """
        Builds the compatible spaces associated with some de Rham complex, and
        sets the spaces as attributes to the underlying :class:`Spaces` object.

        Args:
            family (str): the family of the compatible spaces. This is either
                the horizontal element of the HDiv or HCurl spaces.
            horizontal_degree (int): the horizontal degree of the L2 space.
            vertical_degree (int, optional): the vertical degree of the L2
                space. Defaults to None.
            complex_name (str, optional): optional name to pass to the
                :class:`DeRhamComplex` object to be created. Defaults to None.
        """

        if vertical_degree is not None:
            degree_key = (horizontal_degree, vertical_degree)
            if complex_name is None:
                if horizontal_degree != vertical_degree:
                    complex_name = f'_{horizontal_degree}_{vertical_degree}'
                else:
                    complex_name = f'_{horizontal_degree}'
        else:
            degree_key = horizontal_degree
            if complex_name is None:
                complex_name = f'_{horizontal_degree}'

        if degree_key in self.de_rham_complex.keys():
            raise RuntimeError(f'de Rham complex for degree {degree_key} has '
                               + 'already been created. Cannot create again.')

        # Create the compatible space objects
        de_rham_complex = DeRhamComplex(self.mesh, family, horizontal_degree,
                                        vertical_degree=vertical_degree,
                                        complex_name=complex_name)

        # Set the spaces as attributes of the space container
        self.de_rham_complex[degree_key] = de_rham_complex
        setattr(self, "H1", de_rham_complex.H1)
        setattr(self, "HCurl", de_rham_complex.HCurl)
        setattr(self, "HDiv", de_rham_complex.HDiv)
        setattr(self, "L2", de_rham_complex.L2)
        # Register L2 space as DG also
        setattr(self, "DG", de_rham_complex.L2)
        if hasattr(de_rham_complex, 'theta'):
            setattr(self, "theta", de_rham_complex.theta)

    def build_dg1_equispaced(self):
        """
        Builds the equispaced variant of the DG1 function space, which is used in
        recovered finite element schemes.

        Returns:
            (:class:`FunctionSpace`): the equispaced DG1 function space.
        """

        if self.extruded_mesh:
            cell = self.mesh._base_mesh.ufl_cell().cellname()
            hori_elt = FiniteElement('DG', cell, 1, variant='equispaced')
            vert_elt = FiniteElement('DG', interval, 1, variant='equispaced')
            V_elt = TensorProductElement(hori_elt, vert_elt)
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement('DG', cell, 1, variant='equispaced')

        space = FunctionSpace(self.mesh, V_elt, name='DG1_equispaced')
        setattr(self, 'DG1_equispaced', space)
        return space


class DeRhamComplex(object):
    """Constructs and stores the function spaces forming a de Rham complex."""
    def __init__(self, mesh, family, horizontal_degree, vertical_degree=None,
                 complex_name=None):
        """
        Args:
            mesh (:class:`Mesh`): _description_
            family (str): name of the finite element family to be
                created. Defaults to None.
            horizontal_degree (int): the horizontal degree of the finite element
                space to be created.
            vertical_degree (int, optional): the vertical degree of the
                finite element space to be created. Defaults to None.
            complex_name (str, optional): suffix to give to the spaces created
                in this complex. Defaults to None.
        """

        implemented_families = ["DG", "CG", "RT", "RTF", "RTE", "RTCF", "RTCE",
                                "BDM", "BDMF", "BDME", "BDFM", "BDMCE", "BDMCF"]
        if family not in [None]+implemented_families:
            raise NotImplementedError(f'family {family} either not recognised '
                                      + 'or implemented in Gusto')

        self.mesh = mesh
        self.extruded_mesh = hasattr(mesh, '_base_mesh')
        self.family = family
        self.complex_name = complex_name
        self.build_base_spaces(family, horizontal_degree, vertical_degree)
        self.build_compatible_spaces()

    def build_base_spaces(self, family, horizontal_degree, vertical_degree=None):
        """
        Builds the :class:`FiniteElement` objects for the base mesh.

        Args:
            family (str): the family of the horizontal part of either the HDiv
                or HCurl space.
            horizontal_degree (int): the polynomial degree of the horizontal
                part of the L2 space.
            vertical_degree (int, optional): the polynomial degree of the
                vertical part of the L2 space. Defaults to None.
        """

        if family == 'BDFM':
            # Need a special implementation of base spaces here as it does not
            # fit the same pattern as other spaces
            build_bdfm_base_spaces(self, horizontal_degree, vertical_degree)
            return

        if self.extruded_mesh:
            cell = self.mesh._base_mesh.ufl_cell().cellname()
        else:
            cell = self.mesh.ufl_cell().cellname()

        hdiv_family = hcurl_hdiv_dict[family]
        hcurl_family = hdiv_hcurl_dict[family]

        # horizontal base spaces
        self.base_elt_hori_hdiv = FiniteElement(hdiv_family, cell, horizontal_degree+1)
        if hcurl_family is not None:
            self.base_elt_hori_hcurl = FiniteElement(hcurl_family, cell, horizontal_degree+1)
        self.base_elt_hori_dg = FiniteElement("DG", cell, horizontal_degree)
        self.base_elt_hori_cg = FiniteElement("CG", cell, h1_degree(family, horizontal_degree))

        # vertical base spaces
        if vertical_degree is not None:
            self.base_elt_vert_cg = FiniteElement("CG", interval, vertical_degree+1)
            self.base_elt_vert_dg = FiniteElement("DG", interval, vertical_degree)

    def build_compatible_spaces(self):
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

        if self.extruded_mesh:
            # Horizontal and vertical degrees
            # need specifying separately. Vtheta needs returning.
            Vcg = self.build_h1_space()
            setattr(self, "H1", Vcg)
            Vcurl = self.build_hcurl_space()
            setattr(self, "HCurl", Vcurl)
            Vu = self.build_hdiv_space()
            setattr(self, "HDiv", Vu)
            Vdg = self.build_l2_space()
            setattr(self, "L2", Vdg)
            Vth = self.build_theta_space()
            setattr(self, "theta", Vth)

            return Vcg, Vcurl, Vu, Vdg, Vth

        elif self.mesh.topological_dimension() > 1:
            # 2D: two de Rham complexes (hcurl or hdiv) with 3 spaces
            # 3D: one de Rham complexes with 4 spaces
            # either way, build all spaces
            Vcg = self.build_h1_space()
            setattr(self, "H1", Vcg)
            Vcurl = self.build_hcurl_space()
            setattr(self, "HCurl", Vcurl)
            Vu = self.build_hdiv_space()
            setattr(self, "HDiv", Vu)
            Vdg = self.build_l2_space()
            setattr(self, "L2", Vdg)

            return Vcg, Vcurl, Vu, Vdg

        else:
            # 1D domain, de Rham complex has 2 spaces
            # CG, hdiv and hcurl spaces should be the same
            Vcg = self.build_h1_space()
            setattr(self, "H1", Vcg)
            setattr(self, "HCurl", None)
            setattr(self, "HDiv", Vcg)
            Vdg = self.build_l2_space()
            setattr(self, "L2", Vdg)

            return Vcg, Vdg

    def build_hcurl_space(self):
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
        if hdiv_hcurl_dict[self.family] is None:
            logger.warning('There is no HCurl space for this family. Not creating one')
            return None

        if self.extruded_mesh:
            Vh_elt = HCurl(TensorProductElement(self.base_elt_hori_hcurl,
                                                self.base_elt_vert_cg))
            Vv_elt = HCurl(TensorProductElement(self.base_elt_hori_cg,
                                                self.base_elt_vert_dg))
            V_elt = Vh_elt + Vv_elt
        else:
            V_elt = self.base_elt_hori_hcurl

        return FunctionSpace(self.mesh, V_elt, name='HCurl'+self.complex_name)

    def build_hdiv_space(self):
        """
        Builds and returns the HDiv :class:`FunctionSpace`.

        Returns:
            :class:`FunctionSpace`: the HDiv space.
        """
        if self.extruded_mesh:
            Vh_elt = HDiv(TensorProductElement(self.base_elt_hori_hdiv,
                                               self.base_elt_vert_dg))
            Vt_elt = TensorProductElement(self.base_elt_hori_dg,
                                          self.base_elt_vert_cg)
            Vv_elt = HDiv(Vt_elt)
            V_elt = Vh_elt + Vv_elt
        else:
            V_elt = self.base_elt_hori_hdiv
        return FunctionSpace(self.mesh, V_elt, name='HDiv'+self.complex_name)

    def build_l2_space(self):
        """
        Builds and returns the discontinuous L2 :class:`FunctionSpace`.

        Returns:
            :class:`FunctionSpace`: the L2 space.
        """

        if self.extruded_mesh:
            V_elt = TensorProductElement(self.base_elt_hori_dg, self.base_elt_vert_dg)
        else:
            V_elt = self.base_elt_hori_dg

        return FunctionSpace(self.mesh, V_elt, name='L2'+self.complex_name)

    def build_theta_space(self):
        """
        Builds and returns the 'theta' space.

        This corresponds to the non-Piola mapped space of the vertical component
        of the velocity. The space will be discontinuous in the horizontal but
        continuous in the vertical.

        Raises:
            AssertionError: the mesh is not extruded.

        Returns:
            :class:`FunctionSpace`: the 'theta' space.
        """
        assert self.extruded_mesh, 'Cannot create theta space if mesh is not extruded'

        V_elt = TensorProductElement(self.base_elt_hori_dg, self.base_elt_vert_cg)

        return FunctionSpace(self.mesh, V_elt, name='theta'+self.complex_name)

    def build_h1_space(self):
        """
        Builds the continuous scalar space at the top of the de Rham complex.

        Args:
            name (str, optional): name to assign to the function space. Default
                is "H1".

        Returns:
            :class:`FunctionSpace`: the continuous space.
        """

        if self.extruded_mesh:
            V_elt = TensorProductElement(self.base_elt_hori_cg, self.base_elt_vert_cg)

        else:
            V_elt = self.base_elt_hori_cg

        return FunctionSpace(self.mesh, V_elt, name='H1'+self.complex_name)


def build_bdfm_base_spaces(de_rham_complex, horizontal_degree, vertical_degree=None):
    """
    Builds the :class:`FiniteElement` objects for the de Rham complex when using
    the BDFM space.

    Args:
        de_rham_complex (:class:`DeRhamComplex`): the de Rham complex to set up
            the spaces for.
        horizontal_degree (int): the polynomial degree of the horizontal
            part of the L2 space in the complex.
        vertical_degree (int, optional): the polynomial degree of the vertical
            part of the L2 space in the complex.
    """

    if de_rham_complex.extruded_mesh:
        cell = de_rham_complex.mesh._base_mesh.ufl_cell().cellname()
    else:
        cell = de_rham_complex.mesh.ufl_cell().cellname()

    hdiv_family = 'BDFM'

    # horizontal base spaces
    de_rham_complex.base_elt_hori_hdiv = FiniteElement(hdiv_family, cell, horizontal_degree+1)
    de_rham_complex.base_elt_hori_dg = FiniteElement("DG", cell, horizontal_degree)

    # Add bubble space
    de_rham_complex.base_elt_hori_cg = FiniteElement("CG", cell, horizontal_degree+1)
    de_rham_complex.base_elt_hori_cg += FiniteElement("Bubble", cell, horizontal_degree+2)

    # vertical base spaces
    if de_rham_complex.extruded_mesh and vertical_degree is not None:
        de_rham_complex.base_elt_vert_cg = FiniteElement("CG", interval, vertical_degree+1)
        de_rham_complex.base_elt_vert_dg = FiniteElement("DG", interval, vertical_degree)


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
