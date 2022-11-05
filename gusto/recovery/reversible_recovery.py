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

        # -------------------------------------------------------------------- #
        # Set up the operators for different transformations
        # -------------------------------------------------------------------- #

        # Does recovery by first projecting into broken space then averaging
        self.recoverer = Recoverer(self.q_low, self.q_recovered,
                                   method=self.opts.broken_method,
                                   boundary_method=self.opts.boundary_method)

        # Obtain the recovered field in the higher order space
        self.interp_high = False
        if self.opts.project_high_method == 'recover':
            self.projector_high = Recoverer(self.q_recovered, self.q_rec_high,
                                            method=self.opts.broken_method,
                                            boundary_method=self.opts.boundary_method)
        elif self.opts.project_high_method == 'project':
            self.projector_high = Projector(self.q_recovered, self.q_rec_high)
        elif self.opts.project_high_method == 'interpolate':
            self.projector_high = Interpolator(self.q_recovered, self.q_rec_high)
            self.interp_high = True
        else:
            raise ValueError(f'Method {self.opts.project_high_method} '
                             + 'for projection to higher space not valid')

        # Obtain the correction in the lower order space
        self.interp_low = False
        if self.opts.project_low_method == 'recover':
            # No boundary method as this is not recovery to higher order space
            self.projector_low = Recoverer(self.q_rec_high, self.q_corr_low,
                                           method=self.opts.broken_method,
                                           boundary_method=None)
        elif self.opts.project_low_method == 'project':
            self.projector_low = Projector(self.q_rec_high, self.q_corr_low)
        elif self.opts.project_low_method == 'interpolate':
            self.projector_low = Interpolator(self.q_rec_high, self.q_corr_low)
            self.interp_low = True
        else:
            raise ValueError(f'Method {self.opts.project_low_method} '
                             + 'for projection to lower space not valid')

        # Final injection operator
        # Should identify low order field in higher order space
        self.interp_inj = False
        if self.opts.injection_method == 'recover':
            self.injector = Recoverer(self.q_corr_low, self.q_corr_high,
                                      method=self.opts.broken_method,
                                      boundary_method=self.opts.boundary_method)
        elif self.opts.injection_method == 'project':
            self.injector = Projector(self.q_corr_low, self.q_corr_high)
        elif self.opts.injection_method == 'interpolate':
            self.injector = Interpolator(self.q_corr_low, self.q_corr_high)
            self.interp_inj = True
        else:
            raise ValueError(f'Method {self.opts.injection_method} for injection not valid')

    def project(self):
        self.recoverer.project()
        self.projector_high.interpolate() if self.interp_high else self.projector_high.project()
        self.projector_low.interpolate() if self.interp_low else self.projector_low.project()
        self.q_corr_low.assign(self.q_low - self.q_corr_low)
        self.injector.interpolate() if self.interp_inj else self.injector.project()
        self.q_high.assign(self.q_corr_high + self.q_rec_high)
