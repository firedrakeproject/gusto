from gusto import *
from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
import numpy as np

specified_Vbar = True
analytical_derivatives = True

# set up mesh
Lx = 2
Ly = 2
delta_x = 0.01
nx = int(Lx/delta_x)

mesh = PeriodicRectangleMesh(nx, nx, Lx, Ly, direction="x")

# set up parameters
H = 1.
f = 10
g = 10
parameters = ShallowWaterParameters(H=H, g=g)
Bu = 10
L = sqrt(g*H/(f**2*Bu)) # = 0.1
Ro = 0.1
d_eta = Ro*f**2*L**2/g
dt = 250.

dirname="bickley_jet_eigenmodes"
x, y = SpatialCoordinate(mesh)

output = OutputParameters(dirname=dirname)

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

if specified_Vbar:
    # expression for the background velocity as given
    vexpr = -(g*d_eta/f*L) * (1/cosh(x/L))**2
    Uexpr = Function(Vu).project(as_vector((0., vexpr)))
    u_b.interpolate(Uexpr)
else:
    # use the streamfunction to define a balanced background velocity
    Vpsi = FunctionSpace(mesh, "CG", 2)
    psi = Function(Vpsi)
    w = TestFunction(Vu)
    u_ = TrialFunction(Vu)
    ap = inner(w, u_)*dx
    Lp = inner(w, state.perp(grad(psi)))*dx
    prob = LinearVariationalProblem(ap, Lp, u_b)
    solver = LinearVariationalSolver(prob)
    solver.solve()

# set up function spaces to store the eigenmodes
eigenmodes_real, eigenmodes_imag = Function(W), Function(W)

# set up test and trial functions
velocity, eta = TrialFunctions(W)
w_vector, phi = TestFunctions(W)

# to simplify writing down the weak forms
u = velocity[0]
v = velocity[1]
w = w_vector[0]
tau = w_vector[1]

# define the other functions we need to build the matrices
v_bar = u_b[1]
eta_bar = D_b

# derivatives of V_bar and eta_bar
if analytical_derivatives:
    # derivatives of the specified functions
    dxeta_bar = Function(W.sub(1))
    dxeta_bar_expr = -d_eta/L * (1/cosh(x/L))**2
    dxeta_bar.interpolate(dxeta_bar_expr)
    dxv_bar_expr = (2*g*d_eta)/(f*L**2)*(1/cosh(x/L)**2)*tanh(x/L)
    Ubar = Function(Vu).project(as_vector((0., dxv_bar_expr)))
    dxv_bar = Ubar[1]
else:
    # derivatives calculated using Firedrake
    dxv_bar = v_bar.dx(0)
    dxeta_bar = eta_bar.dx(0)

# set up arrays to store all k's, eigenvectors and eigenvalues
k_list = []
eigenvalue_list = []
eigenmode_list = []
sigma_list = []

# loop over range of k values
#for k in np.arange(0.08, 2.58, 0.08):
for k in np.arange(2.6, 8.6, 0.2):
    print(k)
    eigenmodes_real, eigenmodes_imag = Function(W), Function(W)

    a = w * v_bar * u * dx + tau * 1/Ro * u * dx + tau * dxv_bar * u * dx
    + phi * eta_bar * u.dx(0) * dx + phi * dxeta_bar * u * dx
    + phi * Bu/Ro * u * dx + w * 1/(Ro*k**2) * v * dx + tau * v_bar * v * dx
    + phi * eta_bar * v * dx + phi * Bu/Ro * v * dx
    + w.dx(0) * 1/(Ro*k**2) * eta * dx + tau * 1/Ro * eta * dx
    + phi * v_bar * eta * dx

    m = (u * w + v * tau + eta * phi) * dx

    petsc_a = assemble(a).M.handle
    petsc_m = assemble(m).M.handle

    num_eigenvalues = 1

    opts = PETSc.Options()
    opts.setValue("eps_gen_non_hermitian", None)
    opts.setValue("st_pc_factor_shift_type", "NONZERO")
    opts.setValue("eps_largest_imaginary", None)
    opts.setValue("eps_tol", 1e-6)

    es = SLEPc.EPS().create(comm=COMM_WORLD)
    es.setDimensions(num_eigenvalues)
    es.setOperators(petsc_a, petsc_m)
    es.setFromOptions()
    es.solve()

    nconv = es.getConverged()
    print("Number of converged eigenpairs for k = %f is %f" %(k, nconv))
    outfile = File("eigenmode_%f.pvd"%k)
    if nconv > 0:
        vr, vi = petsc_a.getVecs()
        lam = es.getEigenpair(0, vr, vi)
        with eigenmodes_real.dat.vec as vr:
            with eigenmodes_imag.dat.vec as vi:
                ur, etar = eigenmodes_real.split()
                ui, etai = eigenmodes_imag.split()
                outfile.write(ur, etar, ui, etai)
        k_list.append(k)
        eigenvalue_list.append(lam)
        sigma_list.append(k*np.imag(lam))

# Extract k-value corresponding to the largest growth rate
print("sigma list:")
print(sigma_list)
max_sigma = max(sigma_list)
print("maximum growth rate: %f" %max_sigma)
index = np.argmax(sigma_list)
k_value = k_list[index]
print("k value corresponding to the maximum growth rate: %f" %k_value)

# Save figures
import matplotlib.pyplot as plt

plt.scatter(k_list, np.real(eigenvalue_list))
plt.xlabel('k')
plt.ylabel('c_p')
plt.savefig('cp_plot')

plt.scatter(k_list, sigma_list)
plt.xlabel('k')
plt.ylabel('sigma')
plt.savefig('growth_rate_plot')

