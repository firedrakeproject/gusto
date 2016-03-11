from dcore import *
from firedrake import IcosahedralSphereMesh, Expression, \
    VectorFunctionSpace

refinements = 3  # number of horizontal cells = 20*(4^refinements)

R = 6371220.

u_0 = 20.0  # Maximum amplitude of the zonal wind (m/s)

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements)
global_normal = Expression(("x[0]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                            "x[1]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                            "x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])"))
mesh.init_cell_orientations(global_normal)

# Space for initialising velocity
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)

# Make a vertical direction for the linearised advection
k = Function(W_VectorCG1).interpolate(Expression(("x[0]/pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)","x[1]/pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)","x[2]/pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],0.5)")))

Omega = Function(W_VectorCG1).assign(0.0)

state = ShallowWaterState(mesh, vertical_degree=1, horizontal_degree=1,
                          family="BDM")

g = 9.806
Omega = 7.292e-5
# interpolate initial conditions
# Initial/current conditions
u0, D0 = Function(state.V[0]), Function(state.V[1])
uexpr = Expression(("-u_0*x[1]/R", "u_0*x[0]/R", "0.0"), u_0=u_0, R=R)
Dexpr = Expression("h0 - ((R * Omega * u_0 + u_0*u_0/2.0)*(x[2]*x[2]/(R*R)))/g", h0=2940, R=R, Omega=Omega, u_0=u_0, g=g)

u0.project(uexpr)
D0.interpolate(Dexpr)

state.initialise([u0, D0])

# names of fields to dump
state.fieldlist = ('u', 'D')
