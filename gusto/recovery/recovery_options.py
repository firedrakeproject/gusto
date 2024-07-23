
from firedrake import (interval, FiniteElement, TensorProductElement, FunctionSpace)
from gusto.core.function_spaces import DeRhamComplex
from gusto.core.configuration import RecoveryOptions


class RecoverySpaces():
    """
    Finds or builds necessary spaces to carry out recovery transport for lowest
    and mixed order domains (0,0), (0,1) and  (1,0)
    """

    def __init__(self, domain):

        family = domain.family
        self.domain = domain
        self.mesh = domain.mesh
        # Need spaces from current deRham and a higher order deRham
        self.de_Rham = DeRhamComplex(self.mesh, family,
                                     horizontal_degree=1,
                                     vertical_degree=1,
                                     complex_name='recovery_de_Rham')

        self.extruded_mesh = hasattr(self.mesh, "_base_mesh")
        if self.extruded_mesh:
            self.cell = self.mesh._base_mesh.ufl_cell().cellname()
            self.build_theta_options()
        else:
            self.cell = self.mesh.ufl_cell().cellname()

        self.build_DG_options()
        self.build_HDiv_options()

    def build_DG_options(self):
        # Method to build the DG options for recovery
        DG_embedding_space = self.domain.spaces.DG1_equispaced
        DG_recovered_space = self.domain.spaces.H1

        self.DG_options = RecoveryOptions(embedding_space=DG_embedding_space,
                                          recovered_space=DG_recovered_space)

    def build_theta_options(self):
        # Method to build the theta options for recovery
        DG_hori_ele = FiniteElement('DG', self.cell, 1, variant='equispaced')
        DG_vert_ele = FiniteElement('DG', interval, 2, variant='equispaced')
        CG_hori_ele = FiniteElement('CG', self.cell, 1)
        CG_vert_ele = FiniteElement('CG', interval, 2)

        VDG_ele = TensorProductElement(DG_hori_ele, DG_vert_ele)
        VCG_ele = TensorProductElement(CG_hori_ele, CG_vert_ele)
        VDG_theta = FunctionSpace(self.mesh, VDG_ele)
        VCG_theta = FunctionSpace(self.mesh, VCG_ele)

        self.theta_options = RecoveryOptions(embedding_space=VDG_theta,
                                             recovered_space=VCG_theta)

    def build_HDiv_options(self):
        # Method to build the HDiv options for recovery
        HDiv_embedding_Space = getattr(self.de_Rham, 'HDiv')
        HDiv_recovered_Space = getattr(self.de_Rham, 'HCurl')

        self.HDiv_options = RecoveryOptions(embedding_space=HDiv_embedding_Space,
                                            recovered_space=HDiv_recovered_Space,
                                            injection_method='recover',
                                            project_high_method='project',
                                            project_low_method='project',
                                            broken_method='project')
