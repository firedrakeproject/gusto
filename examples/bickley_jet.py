from gusto import *
from firedrake import (PeriodicRectangleMesh, pi, sin, cos, cosh, sinh,
                       sqrt, exp, TestFunction, TrialFunction,
                       LinearVariationalProblem, LinearVariationalSolver,
                       inner, grad, dx, PCG64, RandomGenerator)

# set up mesh
Lx = 3e6
Ly = 3e6
delta_x = 3e4
nx = int(Lx/delta_x)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction="x")

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

state = State(mesh, dt=dt, output=output, parameters=parameters, diagnostic_fields=[VelocityX(), VelocityY(), CourantNumber()])

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=Constant(f), no_normal_flow_bc_ids=[1,2])

u0 = state.fields("u")
D0 = state.fields("D")

Dexpr = H - d_eta * sinh((y-Ly/2)/L)/cosh((y-Ly/2)/L)

VD = D0.function_space()
Dbackground = Function(VD)
Dbackground.interpolate(Dexpr)

Drandom = Function(VD)
pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)
f_beta = rg.beta(VD, 1.0, 2.0)
amp = max(Dbackground.dat.data)
Drandom.interpolate(0.01*amp + f_beta)
print("Number of random numbers: " + str(len(f_beta.dat.data)))
print(Drandom.dat.data.max())
print(Drandom.dat.data.min())

D0.interpolate(conditional(y<Ly/2+L/10, conditional(y>Ly/2-L/10, Dbackground+Drandom, Dbackground), Dbackground))
#D0.interpolate(conditional(Ly/2 - L/2 < y, conditional(y < Ly/2 + L/2, Dbackground + Drandom, Dbackground), Dbackground))
#D0.interpolate(Dbackground)

Vpsi = FunctionSpace(mesh, "CG", 2)
psi = Function(Vpsi)
psi.interpolate((g/f)*Dbackground) # should this be D0? balance with which D?

Vu = u0.function_space()
w = TestFunction(Vu)
u_ = TrialFunction(Vu)

ap = inner(w, u_)*dx
Lp = inner(w, state.perp(grad(psi)))*dx
prob = LinearVariationalProblem(ap, Lp, u0)
solver = LinearVariationalSolver(prob)
solver.solve()

advected_fields = [ImplicitMidpoint(state, "u"),
                   SSPRK3(state, "D")]

# build time stepper
stepper = CrankNicolson(state, eqns, advected_fields)

#stepper = Timestepper(state, ((eqns, SSPRK3(state)),))
stepper.run(t=0, tmax=240*dt)
