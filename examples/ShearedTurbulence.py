from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    tanh, cosh, cos, pi, SpatialCoordinate
import numpy as np

dt = 0.04
columns = 125  # number of columns
L = 3.29
m = PeriodicIntervalMesh(columns, L)

# build 2D mesh by extruding the base mesh
nlayers = 128  # horizontal layers
H = 1.63 # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'p', 'b']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='KH', dumpfreq=1, dumplist=['u', 'p', 'b', 'bbar'], perturbation_fields=['b'])
diagnostic_fields = [CourantNumber()]
parameters = CompressibleParameters(g=0.981)

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

u0 = state.fields("u")
p0 = state.fields("p")
b0 = state.fields("b")

x = SpatialCoordinate(mesh)
g = parameters.g
z0 = 0.5*H
th0 = Constant(2.41e-6)
h0 = Constant(0.24)
bref = 0.5*g*th0*tanh(2*(x[1]-z0)/h0)
Vb = b0.function_space()
b_b = Function(Vb).interpolate(bref)
incompressible_hydrostatic_balance(state, b_b, p0)
k0 = Constant(0.9/h0)
r = Function(b0.function_space()).assign(Constant(0.0))
r.dat.data[:] += np.random.uniform(low=-1., high=1., size=r.dof_dset.size)
b_pert = 0.5*g*th0/(cosh(2*(x[1]-z0)/h0))**2
a = 0.05
b0.interpolate(bref + a*r*b_pert)

U = Constant(0.00834)
u_base = 0.5*U*tanh(2*(x[1]-z0)/h0)
b = 0.4177
u_pert = 0.5*U/k0*(-cos(2*k0*x[0]/h0) + 2*b*cos(k0*x[1]/h0))*tanh(2*(x[1]-z0)/h0)/cosh(2*(x[1]-z0)/h0)
u0.project(as_vector([u_base + a*u_pert, 0.]))

state.initialise({"u": u0, "p": p0, "b": b0})
state.set_reference_profiles({"b": b_b})

ueqn = EulerPoincare(state, u0.function_space())
supg = True
if supg:
    beqn = SUPGAdvection(state, Vb,
                         supg_params={"dg_direction":"horizontal"},
                         equation_form="advective")
else:
    beqn = EmbeddedDGAdvection(state, Vb,
                               equation_form="advective")
advection_dict = {}
advection_dict["u"] = ThetaMethod(state, u0, ueqn)
advection_dict["b"] = SSPRK3(state, b0, beqn)
linear_solver = IncompressibleSolver(state, L)
forcing = IncompressibleForcing(state)
stepper = Timestepper(state, advection_dict, linear_solver,
                      forcing)
stepper.run(t=0, tmax=7200.)

