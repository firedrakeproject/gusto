from firedrake import *
from gusto import *

# set up parameters
f = 10
g = 10
L = 0.1
Ro = 0.1
Bu = 10
U = f * L * Ro
H = (f**2 * L**2 * Bu)/g  # 1
d_eta = (f * L * U)/g
Rd = sqrt(g * H)/f

# set up mesh
Ly = 2
nx = 100
mesh = PeriodicRectangleMesh(nx, nx, Ly, Ly, direction="x")

parameters = ShallowWaterParameters(H=H, g=g)
dt = 1e-3

dirname = "balanced_bickley_jet_dirichletbcs"

x, y = SpatialCoordinate(mesh)

output = OutputParameters(dirname=dirname, dumpfreq=1)

state = State(mesh, dt=dt, output=output, parameters=parameters, diagnostic_fields=[VelocityX(), VelocityY(), RelativeVorticity()])

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=Constant(f), no_normal_flow_bc_ids = [1, 2])

u0 = state.fields("u")
D0 = state.fields("D")

coordinate = (y - 0.5 * Ly)/L
Dexpr = H - d_eta * (sinh(coordinate)/cosh(coordinate))
VD = D0.function_space()
Dbackground = Function(VD)
Dbackground.interpolate(Dexpr)

amp = Dbackground.at(Ly/2, Ly/2)  # height in the jet

# read height of most unstable mode from eigenmodes calculation
with CheckpointFile("eigenmode.h5", 'r') as afile:
    temp_mesh = afile.load_mesh("rectangle_mesh")
    eta_real = afile.load_function(temp_mesh, "eta_real")

# project unstable mode height on to D field and add it to the background height
Dmode = project(eta_real, VD)
Dnoise = 0.1 * amp * Dmode
D0.interpolate(conditional(y < Ly/2 + L, conditional(y > Ly/2, Dbackground+Dnoise, Dbackground), Dbackground))

# Calculate initial velocity that is in geostrophic balance with the height
Vpsi = FunctionSpace(mesh, "CG", 2)
psi = Function(Vpsi)
psi.interpolate((g/f)*D0)

Vu = u0.function_space()
w = TestFunction(Vu)
u_ = TrialFunction(Vu)

ap = inner(w, u_)*dx
Lp = inner(w, state.perp(grad(psi)))*dx
prob = LinearVariationalProblem(ap, Lp, u0)
solver = LinearVariationalSolver(prob)
solver.solve()

# output the height perturbation
outfile = File("height.pvd")
eta_pert = Function(VD, name="height perturbation")
eta_pert.interpolate(state.fields("D") - Dbackground)
outfile.write(eta_pert, Dmode)

# Timestep the problem
advected_fields = [ImplicitMidpoint(state, "u"),
                   SSPRK3(state, "D")]
stepper = CrankNicolson(state, eqns, advected_fields)
T_i = 1/f
stepper.run(t=0, tmax=5*dt)
