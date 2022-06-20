from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
import matplotlib.pyplot as plt
import numpy as np
from gusto import *

# set up parameters
f = 10
g = 10
L = 0.1
Ro = 0.1
Bu = 10
U = f * L * Ro # 0.1
H = (f**2 * L**2 * Bu)/g # 1
d_eta = (f * L * U)/g # 0.01

# set up mesh
Rd = sqrt(g * H)/f # 0.316
Ly = 2
nx = 100
mesh = IntervalMesh(nx, Ly)

y = SpatialCoordinate(mesh)[0]

coordinate = (y - 0.5 * Ly)/L

# set up spaces
V = FunctionSpace(mesh, "CG", 2)
Z = V*V*V
u, v, eta = TrialFunction(Z)
w, tau, phi = TestFunction(Z)

u_bar, _, eta_bar = Function(Z).split()
u_bar_expr = (g * d_eta)/(f*L) * (1/cosh(coordinate))**2
eta_bar_expr = -d_eta * (sinh(coordinate)/cosh(coordinate))
u_bar.interpolate(u_bar_expr)
eta_bar.interpolate(eta_bar_expr)

# plot functions to check
u_bar, _, eta_bar = Function(Z).split()
# shift axis so that perturbation is at 0
plotting_y = Function(V)
plot_y = plotting_y.interpolate((y - 0.5 * Ly)/Rd).vector()
# u_bar, divided by U
u_bar.interpolate((g * d_eta)/(f*L) * (1/cosh(coordinate))**2)
plotting_u_bar = Function(V)
plot_u_bar = plotting_u_bar.interpolate(u_bar_expr/U).vector()
plt.plot(plot_y, plot_u_bar)
plt.xlim(left=-1.5, right=1.5)
plt.xlabel("y/Rd")
plt.ylabel("u_bar/U")
plt.show()
# eta_bar, divided by d_eta
eta_bar.interpolate(-d_eta * (sinh(coordinate)/cosh(coordinate)))
plotting_eta_bar = Function(V)
plot_eta_bar = plotting_eta_bar.interpolate(eta_bar_expr/d_eta).vector()
plt.plot(plot_y, plot_eta_bar)
plt.xlim(left=-1.5, right=1.5)
plt.xlabel("y/Rd")
plt.ylabel("eta_bar/d_eta")
plt.show()

bc = DirichletBC(Z.sub(1), Constant(0), "on_boundary")

k = 9.738937226128359
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
# vecr, veci = petsc_a.getVecs()

with eigenmodes_real.dat.vec as vecr:
    with eigenmodes_imag.dat.vec as veci:
        lam = es.getEigenpair(0, vecr, veci)
        ur, vr, etar = eigenmodes_real.split()
        ui, vi, etai = eigenmodes_imag.split()

# Non-dimensionalise k by multiplying it by L and c_p by dividing it by U
k = k*L
c_p = np.real(lam)/U

plot(etar)
plt.show()

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
x,y = SpatialCoordinate(mesh)
# Non-dimensionalise x and y by dividing by Rd
# x = x/Rd
# y = y/Rd
x = (x - 0.5 * Ly)/Rd
coordinate = ((y - 0.5 * Ly)/Rd)/L
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

# Plot this eigenmode
# plots are not yet on the same figure
x_array = np.arange(0, Ly, Ly/nx)
y_array = np.arange(0, Ly, Ly/nx)
X, Y = np.meshgrid(x_array, y_array)
u_real_array = u_real.vector().array()
v_real_array = v_real.vector().array()
skip = (slice(None, None, 100))
plt.quiver(u_real_array[skip], v_real_array[skip])
tcf = tricontourf(eta_real, levels=14, cmap="RdBu_r")
cb = plt.colorbar(tcf)
# plt.xlim(left=1.5, right=4.5)
# plt.ylim(bottom=1.5, top=4.5)
# plt.xlabel("x/Rd")
# plt.ylabel("y/Rd")
plt.title("Re(eta)")
plt.show()

# Write this eigenmode out to a file
outfile = File("output.pvd")
outfile.write(u_real, v_real, eta_real)


# Part 2 : Bickley jet with this mode superimposed
new_mesh = PeriodicRectangleMesh(nx, nx, Ly/Rd, Ly/Rd, direction="x")
parameters = ShallowWaterParameters(H=H, g=g)
dt = 250

dirname="June_bickley_jet"
x, y = SpatialCoordinate(new_mesh)

output = OutputParameters(dirname=dirname, dumpfreq=40)

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
velocity_mode = as_vector([u_real_expr, v_real_expr])
print(type(velocity_mode))
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
# advected_fields = [ImplicitMidpoint(state, "u"),
#                    SSPRK3(state, "D")]

# stepper = CrankNicolson(state, eqns, advected_fields)
# T_i = 1/f
# stepper.run(t=0, tmax=40*T_i)
