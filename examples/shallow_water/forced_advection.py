from gusto import *
from firedrake import (PeriodicIntervalMesh, SpatialCoordinate, FunctionSpace,
                       VectorFunctionSpace, conditional, acos, cos, pi, plot,
                       FiniteElement, as_vector, errornorm)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import matplotlib.pyplot as plt

tophat = False
triangle = False
trig = True

u_max = 1
if tophat:
    qmax = 0.7
    qh = 0.2
elif triangle:
    qmax = 0.9
    qh = 0.4
elif trig:
    C0 = 0.6
    K0 = 0.3
Csat = 0.75
Ksat = 0.25

if tophat:
    dirname = "forced_advection_hat"
elif triangle:
    dirname = "forced_advection_triangle"
elif trig:
    dirname = "DG1_withlimiter_forced_advection_trig"

dt = 0.005
dx = 0.05
Lx = 100
nx = int(Lx/dx)
mesh = PeriodicIntervalMesh(nx, Lx)
x = SpatialCoordinate(mesh)[0]
x1 = 0
x2 = Lx/4

output = OutputParameters(dirname=dirname, dumpfreq=100)
diagnostic_fields = [CourantNumber()]

state = State(mesh,
              dt=dt,
              output=output,
              parameters=None,
              diagnostics=None,
              diagnostic_fields=diagnostic_fields)

eltDG = FiniteElement("DG", "interval", 1, variant="equispaced")
VD = FunctionSpace(mesh, eltDG)
Vu = VectorFunctionSpace(mesh, "CG", 1)

# initial moisture profile
if tophat:
    mexpr = conditional(x < x2, conditional(x > x1, qmax, qmax-qh), qmax-qh)
elif triangle:
    mexpr = conditional(x < x2, conditional(x > x1, qmax - 2*qh - (4*qh*(x-(Lx/2)))/Lx, qmax-qh), qmax-qh)
elif trig:
    mexpr = C0 + K0*cos((2*pi*x)/Lx)

# define saturation profile
msat_expr = Csat + (Ksat * cos(2*pi*(x/Lx)))
msat = Function(VD)
msat.interpolate(msat_expr)

# set up advection equation
meqn = AdvectionEquation(state, VD, field_name="water_v", Vu=Vu)
state.fields("u").project(as_vector([u_max]))
state.fields("water_v").project(mexpr)

# define rain variable
r = state.fields("rain", VD)

# exact rainfall profile (analytically)
r_exact = state.fields("r_exact", VD)
if trig:
    lim1 = Lx/(2*pi) * acos((C0 + K0 - Csat)/Ksat)
    lim2 = Lx/2
else:
    lim1 = Lx/(2*pi) * acos((qmax - Csat)/Ksat)
    lim2 = Lx/2
if tophat:
    exact_expr = (pi*Ksat)/2 * sin((2*pi*x)/Lx)
elif triangle:
    coord = (2*pi*x)/Lx
    exact_expr = ((pi*Ksat)/(2*qh) * sin(coord))*(qmax - Csat - Ksat*cos(coord))
elif trig:
    coord = (Ksat*cos(2*pi*x/Lx) + Csat - C0)/K0
    exact_expr = 2*Ksat*sin(2*pi*x/Lx)*acos(coord)
r_expr = conditional(x < lim2, conditional(x > lim1, exact_expr, 0), 0)
r_exact.interpolate(r_expr)

# add instant rain forcing
physics_list = [InstantRain(state, msat)]
# build time stepper
stepper = PrescribedTransport(state,
                              ((meqn,
                                SSPRK3(state, limiter=VertexBasedLimiter(VD))),),
                              physics_list=physics_list)
stepper.run(t=0, tmax=55)

fig, axes = plt.subplots()
plot(r_exact, axes=axes, label='exact solution', color='green')
plot(state.fields("rain"), axes=axes, label='rain after advection', color='red')
plt.title("Rainfall profile after advecting")
plt.legend()
plt.show()

# calculate L2 error norm
r = state.fields("rain")
L2_error = errornorm(r_exact, r)
print(L2_error)
