# This monkey patch is required so that Gusto can redefine the UFL perp.
# It should be removed if the UFL perp gets its own operator
# The main thing to be aware of is if we could now do (and should avoid doing)
# from gusto import _monkey_patch_ufl()
def _monkey_patch_ufl():
    from ufl.algorithms.apply_algebra_lowering import LowerCompoundAlgebra
    def perp(self, o, a):
        from firedrake import as_vector
        return as_vector([-a[1], a[0]])
    LowerCompoundAlgebra.perp = perp
_monkey_patch_ufl()

from gusto.active_tracers import *                   # noqa
from gusto.common_forms import *                     # noqa
from gusto.configuration import *                    # noqa
from gusto.domain import *                           # noqa
from gusto.diagnostics import *                      # noqa
from gusto.diffusion_methods import *                # noqa
from gusto.equations import *                        # noqa
from gusto.fml import *                              # noqa
from gusto.forcing import *                          # noqa
from gusto.initialisation_tools import *             # noqa
from gusto.io import *                               # noqa
from gusto.labels import *                           # noqa
from gusto.limiters import *                         # noqa
from gusto.linear_solvers import *                   # noqa
from gusto.meshes import *                           # noqa
from gusto.physics import *                          # noqa
from gusto.preconditioners import *                  # noqa
from gusto.recovery import *                         # noqa
from gusto.spatial_methods import *                  # noqa
from gusto.time_discretisation import *              # noqa
from gusto.timeloop import *                         # noqa
from gusto.transport_methods import *                # noqa
from gusto.wrappers import *                         # noqa
