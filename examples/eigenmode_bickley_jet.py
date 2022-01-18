from gusto import *
from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
import numpy as np

# set up mesh
Lx = 3e6
Ly = 3e6
delta_x = 3e4
nx = int(Lx/delta_x)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction="x")

# set up parameters
H = 1000.
f = 1e-4
g = 10
parameters = ShallowWaterParameters(H=H, g=g)
Bu = 10
L = sqrt(g*H/(f**2*Bu))
Ro = 0.1
d_eta = Ro*f**2*L**2/g
dt = 250.

dirname="bickley_jet_eigenmodes"
x, y = SpatialCoordinate(mesh)

output = OutputParameters(dirname=dirname)

#timestepping = TimesteppingParameters(dt=dt)

fieldlist = ['u', 'D']

state = State(mesh, output=output, parameters=parameters,
	diagnostic_fields=[VelocityX(), VelocityY()])

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=Constant(f),
	no_normal_flow_bc_ids=[1,2])

u_b = state.fields("u")
D_b = state.fields("D")

Dexpr = H - d_eta * sinh((y-Ly/2)/L)/cosh((y-Ly/2)/L)
D_b.interpolate(Dexpr)

W = eqns.function_space
Vu, VD = W.split()
#print(type(Vu))
#print(type(Vu.sub(0)))
# Trying to extract the v-component space
### 1
#v_space = Vu.sub(1)
### 2
#vector_space = W.sub(0)
#v_space = vector_space.sub(1)
### 3
#vector_space = W.sub(0)
#Vu, Vv = vector_space.split()
### 4
#v_space = Vu[1]
### 5
#v_space = functionspaceimpl.IndexedFunctionSpace(1, W.sub(0), W)

# use the streamfunction to define a balanced background velocity
Vpsi = FunctionSpace(mesh, "CG", 2)
psi = Function(Vpsi)
psi.interpolate((g/f)*D_b)
w = TestFunction(Vu)
u_ = TrialFunction(Vu)
ap = inner(w, u_)*dx
Lp = inner(w, state.perp(grad(psi)))*dx
prob = LinearVariationalProblem(ap, Lp, u_b)
solver = LinearVariationalSolver(prob)
solver.solve()

# set up function spaces to store the eigenmodes - where should these go?
eigenmodes_real, eigenmodes_imag = Function(W.sub(0)), Function(W.sub(0))

# set up test and trial functions
velocity, eta = TrialFunctions(W)
w_vector, phi = TestFunctions(W)

# to simplify writing down the weak forms
u = velocity[0]
v = velocity[1]
w = w_vector[0]
tau = w_vector[1]

# define what we need to build matrices
v_bar = u_b[1]
eta_bar = D_b
dxv_bar = Function(v_space)
dxeta_bar = Function(W.sub(1))
dxv_bar_expr = (2*g*d_eta)/(f*L**2)*(1/cosh(x/L)**2)*tanh(x/L)
dxeta_bar_expr = -d_eta/L * (1/cosh(x/L))**2
dxv_bar.interpolate(dxv_bar_expr)
dxeta_bar.interpolate(dxeta_bar_expr)

# loop over range of k values
for k in np.arange(0.08, 2.58, 0.08):

    print(k)

    a = w * v_bar * u * dx + tau * 1/Ro * u * dx + tau * dxv_bar * u * dx
    + phi * eta_bar * u.dx(0) * dx + phi * dxeta_bar * u * dx
    + phi * Bu/Ro * u * dx + w * 1/(Ro*k**2) * v * dx + tau * v_bar * v * dx
    + phi * eta_bar * v * dx + phi * Bu/Ro * v * dx
    + w.dx(0) * 1/(Ro*k**2) * eta * dx + tau * 1/Ro * eta * dx
    + phi * v_bar * eta * dx

    m = (u * w + v * tau + eta * phi) * dx

    petsc_a = assemble(a).M.handle
    petsc_m = assemble(m).M.handle

    num_eigenvalues = 1 # what should this be?

    opts = PETSc.Options()
    opts.setValue("eps_gen_non_hermitian", None)
    opts.setValue("st_pc_factor_shift_type", "NONZERO")

    es = SLEPc.EPS().create(comm=COMM_WORLD)
    es.setDimensions(num_eigenvalues)
    es.setOperators(petsc_a, petsc_m)
    es.setFromOptions()
    es.solve()

# For each k, check every found eigenvalue and store any that have an imaginary part > 0.
# Save the maximum of these (biggest positive imaginary part) for each k.
# Growth rate is k*eigenvalue.   
