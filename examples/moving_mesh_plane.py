from gusto import *
from firedrake import PeriodicRectangleMesh, Expression, SpatialCoordinate, \
    Constant, as_vector
from math import pi

# setup geometry parameters
nx = 50
ny = 50
L = 10.0
mesh = PeriodicRectangleMesh(nx, ny, L, L)

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber()]

delta_x = L/nx
dt = 0.025
dirname = "mm_dx%s_dt%s" % (delta_x, dt)
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname=dirname, dumpfreq=10)

state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                          family="BDM",
                          timestepping=timestepping,
                          output=output,
                          parameters=parameters,
                          diagnostics=diagnostics,
                          diagnostic_fields = diagnostic_fields,
                          fieldlist=fieldlist)

# interpolate initial conditions
u0, D0 = Function(state.V[0]), Function(state.V[1])
x = SpatialCoordinate(mesh)
Dexpr = Expression("exp(-pow((L/2.-x[1]),2) - pow((L/2.-x[0]),2))", L=L)
D0.interpolate(Dexpr)
state.initialise([u0, D0])

# Coriolis expression
fexpr = Constant(0.0)
V = FunctionSpace(mesh, "CG", 1)
state.f = Function(V).interpolate(fexpr)  # Coriolis frequency (1/s)

advection_dict = {}
advection_dict["D"] = DGAdvection(state, state.V[1], continuity=True)

# build time stepper
vexpr = Expression(("0.0","2*sin(x[1]*pi/L)*sin(2*x[0]*pi/L)*sin(0.5*pi*t)"), t=0, L=L)
Vu = VectorFunctionSpace(mesh, "DG", 1)
u_max = Constant(1.0)
uexpr = as_vector([u_max, 0.0])
uadv = Function(Vu).interpolate(uexpr)
moving_mesh_advection = MovingMeshAdvection(state, advection_dict, vexpr, uadv=uadv)
stepper = MovingMeshAdvectionTimestepper(state, advection_dict, moving_mesh_advection)

stepper.run(t=0, tmax=10.)
