from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector, cos, sin, asin, atan_2
import sys

dt = 900.
day = 24.*60.*60.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 14*day

refinements = 4  # number of horizontal cells = 20*(4^refinements)

R = 6371220.
H = 8000.

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)

fieldlist = ['u', 'D']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='sw_rossby_wave_ll', dumpfreq=24, dumplist_latlon=['D'])
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

# interpolate initial conditions
# Initial/current conditions
u0 = state.fields("u")
D0 = state.fields("D")
Rc = Constant(R)
omega = Constant(7.848e-6)  # note lower-case, not the same as Omega
K = Constant(7.848e-6)
h0 = Constant(H)
g = Constant(parameters.g)
Omega = Constant(parameters.Omega)

x0, y0, z0 = SpatialCoordinate(mesh)
x = Rc*x0/sqrt(x0*x0 + y0*y0 + z0*z0)
y = Rc*y0/sqrt(x0*x0 + y0*y0 + z0*z0)
z = Rc*z0/sqrt(x0*x0 + y0*y0 + z0*z0)

theta = asin(z/Rc)  # latitude
lamda = atan_2(y, x)  # longitude

u_zonal = Rc*omega*cos(theta) + Rc*K*(cos(theta)**3)*(4*sin(theta)**2 - cos(theta)**2)*cos(4*lamda)
u_merid = -Rc*K*4*(cos(theta)**3)*sin(theta)*sin(4*lamda)

cartesian_u_expr = -u_zonal*sin(lamda) - u_merid*sin(theta)*cos(lamda)
cartesian_v_expr = u_zonal*cos(lamda) - u_merid*sin(theta)*sin(lamda)
cartesian_w_expr = u_merid*cos(theta)

uexpr = as_vector((cartesian_u_expr, cartesian_v_expr, cartesian_w_expr))


def Atheta(theta):
    return 0.5*omega*(2*Omega + omega)*cos(theta)**2 + 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 + 26 - 32/(cos(theta)**2))


def Btheta(theta):
    return (2*(Omega + omega)*K/30)*(cos(theta)**4)*(26 - 25*cos(theta)**2)


def Ctheta(theta):
    return 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 - 6)


Dexpr = h0 + (Rc**2)*(Atheta(theta) + Btheta(theta)*cos(4*lamda) + Ctheta(theta)*cos(8*lamda))/g

# Coriolis expression
fexpr = 2*Omega*z/Rc
V = FunctionSpace(mesh, "CG", 1)
f = state.fields("coriolis", V)
f.interpolate(fexpr)  # Coriolis frequency (1/s)

u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 8})
D0.interpolate(Dexpr)

state.initialise({'u': u0, 'D': D0})

ueqn = EulerPoincare(state, u0.function_space())
Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")

advection_dict = {}
advection_dict["u"] = ThetaMethod(state, u0, ueqn)
advection_dict["D"] = SSPRK3(state, D0, Deqn)

linear_solver = ShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state)

# build time stepper
stepper = Timestepper(state, advection_dict, linear_solver,
                      sw_forcing)

stepper.run(t=0, tmax=tmax)
