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
#mesh = IntervalMesh(nx, Ly)
mesh = IntervalMesh(nx, Ly/Rd)

y = SpatialCoordinate(mesh)[0]

#coordinate = (y - 0.5*Ly)/L
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
eta_bar.interpolate(-d_eta * (sinh(coordinate)/cosh(coordinate)))
plot(eta_bar)
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
for n in np.arange(1, 70, 5):
 #for n in np.arange(0.005, 0.16, 0.005): # np.arange(0.00008, 0.019, 0.002):
    k = (2*pi*n*L)/Ly # reversing lengths
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
    #outfile = File("eigenmode_%f.pvd"%k)
    if nconv > 0:
        with eigenmodes_real.dat.vec as vecr:
            with eigenmodes_imag.dat.vec as veci:
                lam = es.getEigenpair(0, vecr, veci)
                ur, vr, etar = eigenmodes_real.split()
                ui, vi, etai = eigenmodes_imag.split()
                #outfile.write(ur, vr, etar, ui, vi, etai)
        ur_list.append(ur)
        ui_list.append(ui)
        vr_list.append(vr)
        vi_list.append(vi)
        etar_list.append(etar)
        etai_list.append(etai)
        eigenvalue_list.append(lam)
        sigma_list.append(k*np.imag(lam)) # dimensionless already
        k_list.append(k * L) # non-dimensional k
        cp_list.append(np.real(lam)/U) # non-dimensional phase speed

# Extract k-value corresponding to the largest growth rate
print("sigma list:")
print(sigma_list)
max_sigma = max(sigma_list)
print("maximum growth rate: %f" %max_sigma)
index = np.argmax(sigma_list)
k = k_list[index]
print("k value corresponding to the maximum growth rate: %f" %k)

# Plot figures
plt.scatter(k_list, cp_list)
plt.xlabel('k')
plt.ylabel('c_p')
plt.show()
#plt.savefig('cp_plot')

plt.scatter(k_list, sigma_list)
plt.xlabel('k')
plt.ylabel('sigma')
plt.show()
#plt.savefig('growth_rate_plot')

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

# Use interp function to get the correct shape for u, v, eta
# u
u_anom = Function(V)
u_anom.interpolate(ur - u_bar)
ur2 = Function(V2)
ur2.dat.data[:] = interp(u_anom, ur2)(X2.dat.data_ro)
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
x,y = SpatialCoordinate(mesh)
# Non-dimensionalise x and y by dividing by Rd
x = x/Rd
y = y/Rd
# u
u_real_expr = ur2 * cos(k*x) - ui2 * sin(k*x)
u_real = Function(V2, name="Re(u)")
u_real.interpolate(u_real_expr)
# v
v_real_expr = -k * vi2 * cos(k*x) - k * vr2 * sin(k*x)
v_real = Function(V2, name="Re(v)")
v_real.interpolate(v_real_expr)
# eta
eta_real_expr = (etar2 * cos(k*x) - etai2 * sin(k*x)) - H
eta_real = Function(V2, name="Re(eta)")
eta_real.interpolate(eta_real_expr)

# Write this eigenmode to a file
outfile = File("eigenmode_%f.pvd"%k)
outfile.write(u_real, v_real, eta_real)

# Plot this eigenmode
# height as contour plot
tcf = tricontourf(eta_real, levels=14, cmap="RdBu_r")
plt.colorbar(tcf)
plt.xlim(left=0, right=3)
plt.ylim(bottom=1.5, top=4.5)
plt.xlabel("x/Rd")
plt.ylabel("y/Rd")
plt.title("Re(eta)")
plt.show()

# velcoity


# Part 2 : Bickley jet with this mode superimposed
new_mesh = PeriodicRectangleMesh(nx, nx, Ly/Rd, Ly/Rd, direction="x")
parameters = ShallowWaterParameters(H=H, g=g)
dt = 250

dirname="June_bickley_jet"
x, y = SpatialCoordinate(new_mesh)

output = OutputParameters(dirname=dirname, dumpfreq=1)

state = State(new_mesh, dt=dt, output=output, parameters=parameters, diagnostic_fields=[VelocityX(), VelocityY()])

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=Constant(f), no_normal_flow_bc_ids=[1,2])

u0 = state.fields("u")
D0 = state.fields("D")

coordinate = (y - 0.5 * (Ly/Rd))/L
Dexpr = H - d_eta * (sinh(coordinate)/cosh(coordinate))
VD = D0.function_space()
Dbackground = Function(VD)
Dbackground.interpolate(Dexpr)
amp = Dbackground.at(Ly/2, Ly/2) # height in the jet

# project unstable mode height on to D field and add it to the background height
Dmode = project(eta_real,  VD)
Dnoise = 0.001 * amp * Dmode
length = Ly/Rd
D0.interpolate(conditional(y<length/2+L/10, conditional(y>length/2-L/10, Dbackground+Dnoise, Dbackground), Dbackground))

# project unstable mode velocity on to U field


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
# advected_fields = [ImplicitMidpoint(state, "u"),
#                    SSPRK3(state, "D")]

# stepper = CrankNicolson(state, eqns, advected_fields)
# T_i = 1/f
# stepper.run(t=0, tmax=10*dt)

