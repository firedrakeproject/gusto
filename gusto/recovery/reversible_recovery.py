"""
This file provides an object for reconstructing a discontinuous field in a
higher-order function space.
"""

from firedrake import (Projector, Function, Interpolator)
from .recovery import Recoverer


class ReversibleRecoverer(object):
    """
    An object for performing a reconstruction of a low-order discontinuous
    field into a higher-order discontinuous space. This uses the recovery
    operator, but with further adjustments to ensure reversibility.

    :arg source_field:      the source field.
    :arg target_field:      the target_field.
    :arg reconstruct_opts:  an object containing the various options for the
                               reconstruction.
    """
    def __init__(self, source_field, target_field, reconstruct_opts):

        self.opts = reconstruct_opts

        # Declare the fields used by the reconstructor
        self.q_low = source_field
        self.q_high = target_field
        self.q_recovered = Function(self.opts.recovered_space)
        self.q_corr_low = Function(source_field.function_space())
        self.q_corr_high = Function(target_field.function_space())
        self.q_rec_high = Function(target_field.function_space())

        # TODO: what if this is a vector element?
        mesh = self.opts.recovered_space.mesh
        rec_elt = self.opts.recovered_space.ufl_element()
        V_broken = FunctionSpace(mesh, BrokenElement(rec_elt))

        # -------------------------------------------------------------------- #
        # Set up the operators for different transformations
        # -------------------------------------------------------------------- #

        # Does recovery by first projecting into broken space then averaging
        self.recoverer = Recoverer(self.q_low, self.q_recovered, VDG=V_broken,
                                   boundary_method=self.opts.boundary_method)

        # Obtain the recovered field in the higher order space
        if self.opts.project_high_method == 'broken':
            self.projector_high = Recoverer(self.q_recovered, self.q_rec_high,
                                            VDG=V_broken, boundary_method=self.opts.boundary_method)
        elif self.opts.project_high_method == 'project':
            self.projector_high = Projector(self.q_recovered, self.q_rec_high)
        elif self.opts.project_high_method == 'interpolate':
            self.projector_high = Interpolator(self.q_recovered, self.q_rec_high)
        else:
            # Surface projection does not make sense here
            raise ValueError(f'Method {self.opts.project_high_method} '+
                             f'for projection to higher space not valid')

        # Obtain the correction in the lower order space
        if self.opts.project_low_method == 'broken':
            # No boundary method as this is not recovery to higher order space
            self.projector_low = Recoverer(self.q_rec_high, self.q_corr_low,
                                           VDG=V_broken, boundary_method=None)
        elif self.opts.project_low_method == 'project':
            self.projector_low = Projector(self.q_rec_high, self.q_corr_low)
        elif self.opts.project_low_method == 'interpolate':
            self.projector_low = Interpolator(self.q_rec_high, self.q_corr_low)
        else:
            raise ValueError(f'Method {self.opts.project_low_method} '+
                             f'for projection to lower space not valid')

        # Final injection operator
        # Should identify low order field in higher order space
        if self.opts.injection_method == 'broken':
            self.injector = Recoverer(self.q_corr_low, self.q_corr_high,
                                      VDG=V_broken, boundary_method=self.opts.boundary_method)
        elif self.opts.injection_method == 'project':
            self.injector = Projector(self.q_corr_low, self.q_corr_high)
        elif self.opts.injection_method == 'interpolate':
            self.injector = Interpolator(self.q_corr_low, self.q_corr_high)
        else:
            raise ValueError(f'Method {self.opts.project_low_method} '+
                             f'for projection to lower space not valid')

    def project(self):
        self.recoverer.project()
        # TODO: can we find a neater way than try / except
        try:
            self.projector_high.project()
        except:
            self.projector_high.interpolate()
        try:
            self.projector_low.project()
        except:
            self.projector_low.interpolate()
        self.q_corr_low.assign(self.q_low - self.q_corr_low)
        try:
            self.injector.project()
        except:
            self.injector.interpolate()

        self.q_high.assign(self.q_corr_high + self.q_rec_high)
