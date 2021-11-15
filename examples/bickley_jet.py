from gusto import *
from firedrake import PeriodicRectangleMesh, pi, sin, cos, cosh,sinh, sqrt

# set up mesh
Lx = 3e6
Ly = 3e6
delta_x = 3e4
nx = int(Lx/delta_x)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction='y')

# set up parameters
H = 1000. # making this choice
f = 1e-4 # making this choice
g = 10 # making this choice
parameters = ShallowWaterParameters(H=H, g=g)
Bu = 10
L = sqrt(g*H/(f**2*Bu))
Ro = 0.1
d_eta = Ro*f**2*L**2/g
dt = 2500

print(g*d_eta/(f*L)**2)
print(g*H/(f*L)**2)
print(L)
print(d_eta)

dirname="bickley_jet"
x, y = SpatialCoordinate(mesh)

output = OutputParameters(dirname=dirname)

state = State(mesh, dt=dt, output=output, parameters=parameters)

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=Constant(f))

u0 = state.fields("u")
D0 = state.fields("D")

uexpr = Constant(0)
vexpr = (-g*d_eta/(f*L)) * (1/cosh((x-Lx/2)/L))**2
Dexpr = H - d_eta * sinh((x-Lx/2)/L)/cosh((x-Lx/2)/L)

u0.project(as_vector((uexpr, vexpr)))
D0.interpolate(Dexpr)

advected_fields = []
advected_fields.append((SSPRK3(state, "u")))
advected_fields.append((SSPRK3(state, "D")))

stepper = Timestepper(state, ((eqns, SSPRK3(state)),))
stepper.run(t=0, tmax=10*dt)
