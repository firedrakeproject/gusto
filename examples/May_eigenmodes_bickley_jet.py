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

# set up spaces
V = FunctionSpace(mesh, "CG", 2)
Z = V*V*V
u, v, eta = TrialFunction(Z)
w, tau, phi = TestFunction(Z)

# plot functions to check
u_bar, _, eta_bar = Function(Z).split()
u_bar.interpolate((g * d_eta)/(f*L) * (1/cosh(coordinate))**2)
plot(u_bar)
plt.show()
eta_bar.interpolate(H - d_eta * (sinh(coordinate)/cosh(coordinate)))
plot(eta_bar)
plt.show()

# boundary conditions: Dirichlet conditions enforcing v = 0 on both ends of the interval
bc = DirichletBC(Z.sub(1), Constant(0), "on_boundary")

# set up arrays to store all k's, eigenvectors and eigenvalues
k_list = []
eigenvalue_list = []
eigenmode_list = []
sigma_list = []

# loop over range of k values
for n in np.arange(0.0005, 0.02, 0.01):
    k = (2*pi*n*Ly)/L
    print(k)
    eigenmodes_real, eigenmodes_imag = Function(Z), Function(Z)

    a = (
        w * u_bar * u * dx
         - tau * 1/(Ro*k**2) * u * dx
         + phi * (eta_bar + Bu/Ro) * u * dx
         + w * (u_bar.dx(0) - 1/Ro) * v * dx
         + tau * u_bar * v * dx
         + phi * (eta_bar.dx(0) * v + eta_bar * v.dx(0) + Bu/Ro * v.dx(0)) * dx
         + w * 1/Ro * eta * dx
         - tau * 1/(Ro*k**2) * eta.dx(0) * dx
         + phi * u_bar * eta * dx
        )

    m = (u * w + v * tau + eta * phi) * dx

    petsc_a = assemble(a).M.handle
    petsc_m = assemble(m, bcs=bc).M.handle
    
    num_eigenvalues = 1

    opts = PETSc.Options()
    opts.setValue("eps_gen_non_hermitian", None)
    opts.setValue("st_pc_factor_shift_type", "NONZERO")
    opts.setValue("eps_type", "lapack")
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

# Plot figures
plt.scatter(k_list, np.real(eigenvalue_list))
plt.xlabel('k')
plt.ylabel('c_p')
plt.show()
#plt.savefig('cp_plot')

plt.scatter(k_list, sigma_list)
plt.xlabel('k')
plt.ylabel('sigma')
plt.show()
#plt.savefig('growth_rate_plot')


def interp(f1, f2):
    def mydata(X):
        Z = np.zeros_like(f2.dat.data[:])
        for i in range(len(X)):
            j = np.where(X1.dat.data_ro==X[i][1])
            Z[i] = f1.dat.data[j]
        return Z
    return mydata


# Get coordinates of interval mesh at degrees of freedom of profile
m1 = V.ufl_domain()
W1 = VectorFunctionSpace(m1, V.ufl_element())
X1 = interpolate(m1.coordinates, W1)

# create 2D mesh
mesh = RectangleMesh(nx, nx, Ly/Rd, Ly/Rd)
V2 = FunctionSpace(mesh, "CG", 2)
W2 = VectorFunctionSpace(mesh, V2.ufl_element())
X2 = interpolate(mesh.coordinates, W2)

u2 = Function(V2)
u2.dat.data[:] = interp(u_bar, u2)(X2.dat.data_ro)
tricontourf(u2)
plt.show()
