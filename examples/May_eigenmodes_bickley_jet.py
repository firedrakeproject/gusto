from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
import numpy as np
import matplotlib.pyplot as plt
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

# set up mesh
Rd = sqrt(g * H)/f
Ly = 2
nx = 100
print(Rd)
print(H)
mesh = IntervalMesh(nx, Ly)
# mesh = IntervalMesh(nx, Ly/Rd)

y = SpatialCoordinate(mesh)[0]

coordinate = (y - 0.5*Ly)/L
# coordinate = (y - 0.5 * (Ly/Rd))/L

# set up spaces
V = FunctionSpace(mesh, "CG", 2)
Z = V*V*V
u, v, eta = TrialFunction(Z)
w, tau, phi = TestFunction(Z)

# plot functions to check
u_bar, _, eta_bar = Function(Z).split()
u_bar_expr = (g * d_eta)/(f*L) * (1/cosh(coordinate))**2
u_bar.interpolate(u_bar_expr)
plot(u_bar)
plt.show()
eta_bar_expr = -d_eta * (sinh(coordinate)/cosh(coordinate))
eta_bar.interpolate(eta_bar_expr)
plot(eta_bar)
plt.show()

# plot scaled function on scaled axes (this should match the paper)
plotting_y = Function(V)
plot_y = plotting_y.interpolate((y - 0.5 * Ly)/Rd).vector()  # shift middle to 0
plotting_u_bar = Function(V)
plot_u_bar = plotting_u_bar.interpolate(u_bar_expr/U).vector()
plotting_eta_bar = Function(V)
plot_eta_bar = plotting_eta_bar.interpolate(eta_bar_expr/d_eta).vector()
plt.plot(plot_y, plot_u_bar)
plt.xlim(left=-1.5, right=1.5)
plt.xlabel("y/Rd")
plt.ylabel("u_bar/U")
plt.show()
plt.plot(plot_y, plot_eta_bar)
plt.xlim(left=-1.5, right=1.5)
plt.xlabel("y/Rd")
plt.ylabel("eta_bar/d_eta")
plt.show()


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
# for n in np.arange(1, 64, 2):
for n in np.arange(1, 70, 5):
# for n in np.arange(0.005, 0.16, 0.005): np.arange(0.00008, 0.019, 0.002):
    k = (2*pi*n*L)/Ly
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
    print("Number of converged eigenpairs for k = %f is %f" % (k, nconv))
    if nconv > 0:
        with eigenmodes_real.dat.vec as vecr:
            with eigenmodes_imag.dat.vec as veci:
                lam = es.getEigenpair(0, vecr, veci)
                ur, vr, etar = eigenmodes_real.split()
                ui, vi, etai = eigenmodes_imag.split()
                # outfile.write(ur, vr, etar, ui, vi, etai)
        ur_list.append(ur)
        ui_list.append(ui)
        vr_list.append(vr)
        vi_list.append(vi)
        etar_list.append(etar)
        etai_list.append(etai)
        eigenvalue_list.append(lam)
        sigma_list.append(k*np.imag(lam))  # dimensionless already
        # k_list.append(k) # dimensional k
        # cp_list.append(np.real(lam)) # dimensional phase speed
        k_list.append(k * L)  # non-dimensional k
        cp_list.append(np.real(lam)/U)  # non-dimensional phase speed

# Extract k-value corresponding to the largest growth rate
print("sigma list:")
print(sigma_list)
max_sigma = max(sigma_list)
print("maximum growth rate: %f" % max_sigma)
index = np.argmax(sigma_list)
k = k_list[index]
print("k value corresponding to the maximum growth rate: %f" % k)

# Plot figures
plt.scatter(k_list, cp_list)
plt.xlabel('k')
plt.ylabel('c_p')
plt.show()
# plt.savefig('cp_plot')

plt.scatter(k_list, sigma_list)
plt.xlabel('k')
plt.ylabel('sigma')
plt.show()
# plt.savefig('growth_rate_plot')

# Extract fastest-growing eigenmode
ur = ur_list[index]
ui = ui_list[index]
vr = vr_list[index]
vi = vi_list[index]
etar = etar_list[index]
etai = etai_list[index]

plot(etar)
plt.show()


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
mesh = RectangleMesh(nx, nx, Ly, Ly)
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
eta_real = Function(V2, name="Re(eta)")
eta_real.interpolate(eta_real_expr)

# Write this eigenmode to a file
outfile = File("eigenmode_%f.pvd" % k)
outfile.write(u_real, v_real, eta_real)

# Plot this eigenmode
# height as contour plot
tcf = tricontourf(eta_real, levels=14, cmap="RdBu_r")
cb = plt.colorbar(tcf)
scaled_lim1 = Ly/2 - (1.5 * Rd)
scaled_lim2 = Ly/2 + (1.5 * Rd)
plt.xlim(left=scaled_lim1, right=scaled_lim2)
plt.ylim(bottom=scaled_lim1, top=scaled_lim2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Re(eta)")
plt.show()

# # velcoity
# x_array = np.arange(0, Ly/Rd, Ly/nx)
# y_array = np.arange(0, Ly/Rd, Ly/nx)
# X, Y = np.meshgrid(x_array, y_array)
# u_real_array = u_real.vector().array()
# v_real_array = v_real.vector().array()
# plt.quiver(u_real_array, v_real_array)
# plt.show()


# Part 2 : Bickley jet with this mode superimposed
new_mesh = PeriodicRectangleMesh(nx, nx, Ly, Ly, direction="x")
parameters = ShallowWaterParameters(H=H, g=g)
dt = 1e-3

dirname = "June_bickley_jet"
x, y = SpatialCoordinate(new_mesh)

output = OutputParameters(dirname=dirname, dumpfreq=1)

state = State(new_mesh, dt=dt, output=output, parameters=parameters, diagnostic_fields=[VelocityX(), VelocityY()])

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=Constant(f), no_normal_flow_bc_ids=[1, 2])

u0 = state.fields("u")
D0 = state.fields("D")

coordinate = (y - 0.5 * Ly)/L
Dexpr = H - d_eta * (sinh(coordinate)/cosh(coordinate))
VD = D0.function_space()
Dbackground = Function(VD)
Dbackground.interpolate(Dexpr)
amp = Dbackground.at(Ly/2, Ly/2)  # height in the jet

# project unstable mode height on to D field and add it to the background height
Dmode = project(eta_real, VD)
Dnoise = 0.1 * amp * Dmode
D0.interpolate(conditional(y < Ly/2 + L, conditional(y > Ly/2, Dbackground+Dnoise, Dbackground), Dbackground))
D0.interpolate(Dbackground + Dnoise)


# project unstable mode velocity on to u field
velocity_mode = as_vector(([u_real_expr, v_real_expr]))
print(type(([u_real_expr, v_real_expr])))
print(type([u_real_expr, v_real_expr]))
u0.project(velocity_mode)


# Calculate initial velocity that is in geostrophic balance with the height
Vpsi = FunctionSpace(new_mesh, "CG", 2)
psi = Function(Vpsi)
psi.interpolate((g/f)*Dbackground)

Vu = u0.function_space()
w = TestFunction(Vu)
u_ = TrialFunction(Vu)

ap = inner(w, u_)*dx
Lp = inner(w, state.perp(grad(psi)))*dx
prob = LinearVariationalProblem(ap, Lp, u0)
solver = LinearVariationalSolver(prob)
solver.solve()


# Timestep the problem
advected_fields = [ImplicitMidpoint(state, "u"),
                   SSPRK3(state, "D")]

stepper = CrankNicolson(state, eqns, advected_fields)
T_i = 1/f
stepper.run(t=0, tmax=5*dt)
