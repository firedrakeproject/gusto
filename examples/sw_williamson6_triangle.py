from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector, VectorFunctionSpace, cos, sin
import sys

dt = 120.
day = 24.*60.*60.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 14*day

refinements = 5  # number of horizontal cells = 20*(4^refinements)

R = 6371220.
H = 8000.

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)

fieldlist = ['u', 'D']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='sw_rossby_wave_ll', dumpfreq=1, dumplist_latlon=['D'])
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber()]

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

g = parameters.g
Omega = parameters.Omega

# interpolate initial conditions
# Initial/current conditions
u0 = state.fields.u
D0 = state.fields.D
x = SpatialCoordinate(mesh)
R = Constant(R)
V = FunctionSpace(mesh, "CG", 2)
phi = Function(V).interpolate(Expression("atan2(x[1],x[0])"))
lambda0 = Function(V).interpolate(Expression("asin(x[2]/R)", R=R))
omega = Constant(7.848e-6)
K = omega
uexpr = as_vector([R*omega*cos(lambda0) + R*K*cos(lambda0)**3*(4*sin(lambda0)**2 - cos(lambda0)**2)*cos(4*phi), -4*R*K*cos(lambda0)**3*sin(lambda0)*sin(4*phi), 0.0])
h0 = Constant(H)
Omega = Constant(parameters.Omega)
g = Constant(parameters.g)
Dexpr = h0 + R**2/g*(0.5*omega*(2*Omega+omega)*cos(lambda0)**2 + 0.25*K**2*cos(lambda0)**8*(5*cos(lambda0)**2 + 26 - 32/(cos(lambda0)**2)) + ((Omega+omega)*K/15.*cos(lambda0)**4*(26 - 25*cos(lambda0)**2))*cos(4*phi) + 0.25*K**2*cos(lambda0)**8*(5*cos(lambda0)**2-6)*cos(8*phi))
# Coriolis expression
fexpr = 2*Omega*x[2]/R
V = FunctionSpace(mesh, "CG", 1)
state.f = Function(V).interpolate(fexpr)  # Coriolis frequency (1/s)

VX = VectorFunctionSpace(mesh, "Lagrange", 1)
u_init = Function(VX).interpolate(uexpr)
u0.project(u_init)
D0.interpolate(Dexpr)

state.initialise({'u': u0, 'D': D0})
ueqn = EulerPoincare(state, u0.function_space())
Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
advection_dict = {}
advection_dict["u"] = NoAdvection(state, u0, None)
advection_dict["D"] = SSPRK3(state, D0, Deqn)

linear_solver = ShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state)

# build time stepper
stepper = Timestepper(state, advection_dict, linear_solver,
                      sw_forcing)

stepper.run(t=0, tmax=tmax)
