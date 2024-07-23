
from firedrake import (interval, FiniteElement, TensorProductElement, FunctionSpace)
from gusto import (RecoveryOptions)


class RecoverySpaces():

    def __init__(self, domain):

        self.domain = domain
        self.mesh = domain.mesh
        self.extruded_mesh = hasattr(self.mesh, "_base_mesh")

        if self.extruded_mesh:
            self.cell = self.mesh._base_mesh.ufl_cell().cellname()
            self.build_theta_options()
        else:
            self.cell = self.mesh.ufl_cell().cellname()

        self.h_degree = domain.horizontal_degree
        self.v_degree = domain.vertical_degree

        # Call methods to find / build the various spaces
        self.build_DG_options()
        self.build_HDiv_options()

    def build_DG_options(self):
        DG_embedding_space = self.domain.spaces.DG1_equispaced
        #TODO the H1 spaces seems to be what we want but im unsure if this is true for triangular cells
        DG_recovered_space = self.domain.spaces.H1

        self.DG_options = RecoveryOptions(embedding_space=DG_embedding_space,
                                          recovered_space=DG_recovered_space)

    def build_theta_options(self):
        # TODO I Dont think these spaces are built anytwhere so have to build them here
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

        HDiv_embedding_Space = getattr(self.domain.spaces, f'HDiv')
        HDiv_recovered_Space = getattr(self.domain.spaces, f'HCurl')
        # TODO Do we want a check if the spaces exist with an option to build

        self.HDiv_options = RecoveryOptions(embedding_space=HDiv_embedding_Space,
                                            recovered_space=HDiv_recovered_Space,
                                            injection_method='recover',
                                            project_high_method='project',
                                            project_low_method='project',
                                            broken_method='project')
