from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector, VectorFunctionSpace, cos, sin

refinements = 5  # number of horizontal cells = 20*(4^refinements)

R = 6371220.
H = 8000.
day = 24.*60.*60.

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)

fieldlist = ['u', 'D']
timestepping = TimesteppingParameters(dt=120.)
output = OutputParameters(dirname='sw_rossby_wave', dumpfreq=1, dumplist_latlon=['D', 'PotentialVorticity', 'Vorticity', 'Divergence'])
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber(), Divergence(), Vorticity(), PotentialVorticity()]

state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
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
u0, D0 = Function(state.V[0]), Function(state.V[1])
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

state.initialise([u0, D0])
advection_list = []
velocity_advection = EulerPoincareForm(state, state.V[0])
advection_list.append((velocity_advection, 0))
D_advection = DGAdvection(state, state.V[1], continuity=True)
advection_list.append((D_advection, 1))

linear_solver = ShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state)

# build time stepper
stepper = Timestepper(state, advection_list, linear_solver,
                      sw_forcing)

stepper.run(t=0, tmax=14*day)
