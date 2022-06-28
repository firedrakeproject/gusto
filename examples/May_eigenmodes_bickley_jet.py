from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
import numpy as np
from gusto import *

# set up parameters
f = 10
g = 10
L = 0.1
Ro = 0.1
Bu = 10
U = f * L * Ro
H = (f**2 * L**2 * Bu)/g
d_eta = (f * L * U)/g
Rd = sqrt(g * H)/f

# set up mesh
Ly = 2
nx = 100
mesh = IntervalMesh(nx, Ly)

y = SpatialCoordinate(mesh)[0]

coordinate = (y - 0.5*Ly)/Ly

# set up spaces
V = FunctionSpace(mesh, "CG", 2)
Z = V*V*V
u, v, eta = TrialFunction(Z)
w, tau, phi = TestFunction(Z)

# height and velocity in the jet
u_bar, _, eta_bar = Function(Z).split()
u_bar_expr = (g * d_eta)/(f*L) * (1/cosh(coordinate))**2
u_bar.interpolate(u_bar_expr)
eta_bar_expr = -d_eta * (sinh(coordinate)/cosh(coordinate))
eta_bar.interpolate(eta_bar_expr)

# boundary conditions: Dirichlet conditions enforcing v = 0 on both ends of the interval
bc = DirichletBC(Z.sub(1), Constant(0), "on_boundary")

# set up arrays to store all k's, eigenvectors and eigenvalues
k_list = []
eigenvalue_list = []
eigenmode_list = []
sigma_list = []
ur_list = []
ui_list = []
vr_list = []
vi_list = []
etar_list = []
etai_list = []
cp_list = []

# loop over range of k values
for n in np.arange(1, 70, 5):
    k = (2*pi*n*L)/Ly
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
    if nconv > 0:
        with eigenmodes_real.dat.vec as vecr:
            with eigenmodes_imag.dat.vec as veci:
                lam = es.getEigenpair(0, vecr, veci)
                ur, vr, etar = eigenmodes_real.split()
                ui, vi, etai = eigenmodes_imag.split()
        ur_list.append(ur)
        ui_list.append(ui)
        vr_list.append(vr)
        vi_list.append(vi)
        etar_list.append(etar)
        etai_list.append(etai)
        eigenvalue_list.append(lam)
        sigma_list.append(k*np.imag(lam))  # dimensionless already
        k_list.append(k * L)  # non-dimensional k
        cp_list.append(np.real(lam)/U)  # non-dimensional phase speed

# Extract k-value corresponding to the largest growth rate
max_sigma = max(sigma_list)
index = np.argmax(sigma_list)
k = k_list[index]

# Plot figures
# plt.scatter(k_list, cp_list)
# plt.xlabel('k')
# plt.ylabel('c_p')
# plt.show()

# plt.scatter(k_list, sigma_list)
# plt.xlabel('k')
# plt.ylabel('sigma')
# plt.show()

# Extract fastest-growing eigenmode
ur = ur_list[index]
ui = ui_list[index]
vr = vr_list[index]
vi = vi_list[index]
etar = etar_list[index]
etai = etai_list[index]


def interp(f1, f2):
    def mydata(X):
        Z = np.zeros_like(f2.dat.data[:])
        for i in range(len(X)):
            j = np.where(X1.dat.data_ro == X[i][1])
            Z[i] = f1.dat.data[j]
        return Z
    return mydata


# Get coordinates of interval mesh at degrees of freedom of profile
m1 = V.ufl_domain()
W1 = VectorFunctionSpace(m1, V.ufl_element())
X1 = interpolate(m1.coordinates, W1)

# create 2D mesh
mesh = RectangleMesh(nx, nx, Ly, Ly, name="rectangle_mesh")
V2 = FunctionSpace(mesh, "CG", 2)
W2 = VectorFunctionSpace(mesh, V2.ufl_element())
X2 = interpolate(mesh.coordinates, W2)

# Use interp function to get the correct shape for u, v, eta
# u
ur2 = Function(V2)
ur2.dat.data[:] = interp(ur, ur2)(X2.dat.data_ro)
ui2 = Function(V2)
ui2.dat.data[:] = interp(ui, ui2)(X2.dat.data_ro)
# v
vr2 = Function(V2)
vr2.dat.data[:] = interp(vr, vr2)(X2.dat.data_ro)
vi2 = Function(V2)
vi2.dat.data[:] = interp(vi, vi2)(X2.dat.data_ro)
# eta
etar2 = Function(V2)
etar2.dat.data[:] = interp(etar, etar2)(X2.dat.data_ro)
etai2 = Function(V2)
etai2.dat.data[:] = interp(etai, etai2)(X2.dat.data_ro)

# Multiply u, v, eta by the exponential term, retaining only the real part
x, y = SpatialCoordinate(mesh)
# Non-dimensionalise x and y by dividing by L
x = x/L
y = y/L
coordinate = (y - 0.5 * Ly)/L
# u minus background jet
u_real_expr = ur2 * cos(k*x) - ui2 * sin(k*x) - (((g * d_eta)/(f*L) * (1/cosh(coordinate))**2))
u_real = Function(V2, name="Re(u)")
u_real.interpolate(u_real_expr)
# v
v_real_expr = -k * vi2 * cos(k*x) - k * vr2 * sin(k*x)
v_real = Function(V2, name="Re(v)")
v_real.interpolate(v_real_expr)
# eta minus background jet
eta_real_expr = (etar2 * cos(k*x) - etai2 * sin(k*x)) - (- d_eta * (sinh(coordinate)/cosh(coordinate)))
eta_real = Function(V2, name="eta_real")
eta_real.interpolate(eta_real_expr)

# Write this eigenmode to a file
outfile = File("eigenmode_%f.pvd" % k)
outfile.write(eta_real)
with CheckpointFile("eigenmode.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(eta_real)

# Plot this eigenmode
# height as contour plot
# tcf = tricontourf(eta_real, levels=14, cmap="RdBu_r")
# cb = plt.colorbar(tcf)
# scaled_lim1 = Ly/2 - (1.5 * Rd)
# scaled_lim2 = Ly/2 + (1.5 * Rd)
# plt.xlim(left=scaled_lim1, right=scaled_lim2)
# plt.ylim(bottom=scaled_lim1, top=scaled_lim2)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Re(eta)")
# plt.show()
