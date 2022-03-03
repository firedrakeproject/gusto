from gusto import *

# setup shallow water parameters
R = 6371220.
h0 = 1000.

parameters = ShallowWaterParameters(H=h0)

dirname = "sw_rexi"
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
eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr)

# interpolate initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
rc = R/3.
c = conditional(
    R*acos(Min(abs((x[2]*R)/(R*R)), abs(1.0))) < rc,
    50.*h0*(1 + cos(pi*R*acos(Min(abs((x[2]*R)/(R*R)),
                                  abs(1.0)))/rc)), 0.0)
