from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector
from math import pi
import sys

day = 24.*60.*60.
dt = 3000.

# setup shallow water parameters
R = 6371220.
H = 4000.

# setup input that doesn't change with ref level or dt
parameters = MoistShallowWaterParameters(H=H)

dirname = "moist_sw"
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=3)
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
