from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    tanh, cos, pi, SpatialCoordinate

dt = 0.005
columns = 40  # number of columns
L = 0.25
m = PeriodicIntervalMesh(columns, L)

# build 2D mesh by extruding the base mesh
nlayers = 50  # horizontal layers
H = 0.36 # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'p', 'b']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='RayleighTaylor', dumpfreq=1, dumplist=['u', 'p', 'b', 'bbar'], perturbation_fields=['b'])
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
bref = 10*g*(1.+0.05*(1. - tanh(100.*(x[1]-z0))))
Vb = b0.function_space()
b_b = Function(Vb).interpolate(bref)
incompressible_hydrostatic_balance(state, b_b, p0)
lmbda = L
h0 = 0.01*lmbda
k = Constant(2*pi/lmbda)
delta = h0*cos(k*x[0])
b_perturbed = 10*g*(1.+0.05*(2. - tanh(100.*(x[1]-z0+delta))))
b0.interpolate(b_perturbed)

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
stepper.run(t=0, tmax=2.)

