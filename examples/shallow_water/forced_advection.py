from gusto import *
from firedrake import (PeriodicIntervalMesh, SpatialCoordinate, FunctionSpace,
                       VectorFunctionSpace, conditional, acos, cos, pi, plot,
                       FiniteElement, as_vector)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import matplotlib.pyplot as plt

# set up mesh
Lx = 100
delta = 0.05
nx = int(Lx/delta)

mesh = PeriodicIntervalMesh(nx, Lx)
x = SpatialCoordinate(mesh)[0]

tophat = False
triangle = False
trig = True

if tophat:
    dirname = "forced_advection_hat"
elif triangle:
    dirname = "forced_advection_triangle"
elif trig:
    dirname = "forced_advection_trig"

output = OutputParameters(dirname=dirname, dumpfreq=100)

diagnostic_fields = [CourantNumber()]

# set up parameters
dt = 0.005
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
x1 = 0
x2 = Lx/4

state = State(mesh,
              dt=dt,
              output=output,
              parameters=None,
              diagnostics=None,
              diagnostic_fields=diagnostic_fields)

# set up function spaces
eltDG = FiniteElement("DG", "interval", 0, variant="equispaced")
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
meqn = AdvectionEquation(state, VD, field_name="m", Vu=Vu)
state.fields("u").project(as_vector([u_max]))
state.fields("m").project(mexpr)

# define rain variable
rain = state.fields("r", VD)
rain.project(Constant(0.))

# exact rainfall profile (analytically)
r_exact = Function(VD)
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
    coord = (Ksat*cos(2*pi*x/Lx) + Csat - C0)/Ksat
    exact_expr = 2*Ksat*sin(2*pi*x/Lx)*acos(coord)
r_expr = conditional(x < lim2, conditional(x > lim1, exact_expr, 0), 0)
r_exact.interpolate(r_expr)

# plot initial set-up
fig, axes = plt.subplots()
plot(msat, axes=axes, label='m_sat', color='black')
plot(state.fields("m"), axes=axes, label='m_0', color='blue')
plot(r_exact, axes=axes, label='r(x)', color='green')
axes.legend(loc='lower right')
plt.title('Saturation curve, initial moisture profile and analytical rainfall profile')
plt.show()

# set up moisture as a field to be transported
transported_fields = [SSPRK3(state, "m", limiter=VertexBasedLimiter(VD)),
                      ImplicitMidpoint(state, "u")]

# add instant rain forcing
physics_list = [InstantRain(state, msat)]


# prescribe velocity for transport
def transport_u(t):
    return state.fields("u")


# build time stepper
stepper = PrescribedTransport(state, ((meqn, SSPRK3(state)),),
                              physics_list=physics_list,
                              prescribed_transporting_velocity=transport_u)

stepper.run(t=0, tmax=11000*dt)

fig, axes = plt.subplots()
plot(r_exact, axes=axes, label='exact solution', color='green')
plot(state.fields("r"), axes=axes, label='rain after advection', color='red')
plt.title("Rainfall profile after advecting")
plt.legend()
plt.show()
