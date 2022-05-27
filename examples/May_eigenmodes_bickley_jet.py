from firedrake import * 
from firedrake.petsc import PETSc
from slepc4py import SLEPc
import numpy as np
import matplotlib.pyplot as plt

# set up parameters
f = 1
g = 1
L = 0.1
Ro = 0.1
Bu = 10
U = f * L * Ro
H = (f**2 * L**2 * Bu)/g
d_eta = (f * L * U)/g

# set up mesh
Ly = 2
nx = 100
Rd = sqrt(g * H)/f
print(Rd)
print(H)
mesh = IntervalMesh(nx, Ly/Rd)

y = SpatialCoordinate(mesh)[0]

coordinate = (y - 0.5 * (Ly/Rd))/L

u_bar = (g * d_eta)/(f*L) * (1/cosh(coordinate))**2
eta_bar = H - d_eta * (sinh(coordinate)/cosh(coordinate))

# set up spaces
V = FunctionSpace(mesh, "CG", 2)
Z = V*V*V
u, v, eta = TrialFunction(Z)
w, tau, phi = TestFunction(Z)

# plot functions to check
plot_u_bar = Function(V)
plot_u_bar.interpolate(u_bar)
plot(plot_u_bar)
plt.show()
plot_eta_bar = Function(V)
plot_eta_bar.interpolate(eta_bar)
plot(plot_eta_bar)
plt.show()

# boundary conditions: Dirichlet conditions enforcing u = 0 on both ends of the interval
bc = DirichletBC(Z.sub(0), 0, "on_boundary")

# set up arrays to store all k's, eigenvectors and eigenvalues
k_list = []
eigenvalue_list = []
eigenmode_list = []
sigma_list = []

# loop over range of k values
for n in np.arange(0.001, 0.02, 0.001):
    k = (2*pi*n*Ly)/L
    print(k)
    eigenmodes_real, eigenmodes_imag = Function(Z), Function(Z)

    a = w * u_bar * u * dx
    - tau * 1/(Ro*k**2) * u * dx
    + phi * (eta_bar + Bu/Ro) * u * dx
    + w * (u_bar.dx(0)- 1/Ro) * v * dx
    + tau * u_bar * v * dx
    + phi * (eta_bar.dx(0) + eta_bar * v.dx(0) + Bu/Ro * v.dx(0)) * dx
    + w * 1/Ro * eta * dx
    - tau * 1/(Ro*k**2) * eta.dx(0) * dx
    + phi * u_bar * eta * dx

    m = (u * w + v * tau + eta * phi) * dx

    petsc_a = assemble(a).M.handle
    petsc_m = assemble(m, bcs=bc).M.handle

    num_eigenvalues = 1

    opts = PETSc.Options()
    opts.setValue("eps_gen_non_hermitian", None)
    opts.setValue("st_pc_factor_shift_type", "NONZERO")
    opts.setValue("eps_largest_imaginary", None)
    opts.setValue("eps_tol", 1e-10)

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
                ur, vr, etar = eigenmodes_real.split()
                ui, vi, etai = eigenmodes_imag.split()
                outfile.write(ur, vr, etar, ui, vi, etai)
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
