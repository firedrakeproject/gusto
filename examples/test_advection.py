from gusto import *
from firedrake import PeriodicIntervalMesh, \
    ExtrudedMesh, Expression, SpatialCoordinate, \
    as_vector, VectorFunctionSpace, sin
from math import pi

output = OutputParameters(dirname="TGtest", dumplist=["f"], dumpfreq=1)
nlayers = 25  # horizontal layers
columns = 25  # number of columns
L = 1.0
m = PeriodicIntervalMesh(columns, L)

H = 1.0  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
dt = 0.01
tmax = 2.5

# Spaces for initialising k and z
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)

# vertical coordinate and normal
z = Function(W_CG1).interpolate(Expression("x[1]"))
k = Function(W_VectorCG1).interpolate(Expression(("0.","1.")))

fieldlist = ['u','rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)
parameters = CompressibleParameters()
diagnostic_fields = [CourantNumber()]

state = CompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                          family="CG",
                          z=z, k=k,
                          timestepping=timestepping,
                          output=output,
                          parameters=parameters,
                          diagnostic_fields=diagnostic_fields,
                          fieldlist=fieldlist)

uexpr = as_vector([1.0, 0.0])

space = FunctionSpace(mesh, "CG", 2)
f = Function(space, name='f')
x = SpatialCoordinate(mesh)
fexpr = sin(2*pi*x[0])*sin(2*pi*x[1])

# interpolate initial conditions
u0 = Function(state.V[0], name="velocity")
u0.project(uexpr)
f.interpolate(fexpr)
state.field_dict["f"] = f
state.initialise([u0])

fequation = AdvectionEquation(state, f.function_space(), continuity=False, ibp_twice=False)
f_advection = TaylorGalerkin(state, f, fequation)

advection_dict = {}
advection_dict["f"] = f_advection
timestepper = AdvectionTimestepper(state, advection_dict)

timestepper.run(0, tmax)
