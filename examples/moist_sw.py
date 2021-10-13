from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector
from math import pi
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 4000., 4: 2000., 5: 1000., 6: 500.}
    tmax = 5*day

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    dirname = "sw_W2_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    global_normal = x
    mesh.init_cell_orientations(x)

    output = OutputParameters(dirname=dirname)

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters)

    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    eqns = MoistShallowWaterEquations(state, "BDM", 1, fexpr=fexpr)
