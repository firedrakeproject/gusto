from gusto import *
from firedrake import (PeriodicIntervalMesh, SpatialCoordinate, FunctionSpace,
                       VectorFunctionSpace, conditional, acos, cos, pi, plot,
                       FiniteElement, as_vector, errornorm)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter

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
if trig:
    tmax = 85
else:
    tmax = 55

if tophat:
    dirname = "forced_advection_hat"
elif triangle:
    dirname = "forced_advection_triangle"
elif trig:
    dirname = "forced_advection_trig"

dt = 0.005
delta_x = 0.05
Lx = 100
nx = int(Lx/delta_x)
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
rain = Rain(space='tracer', transport_eqn=TransportEquationType.no_transport)
meqn = ForcedAdvectionEquation(state, VD, field_name="water_v", Vu=Vu,
                               active_tracers=[rain])
state.fields("u").project(as_vector([u_max]))
state.fields("water_v").project(mexpr)

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
physics_schemes = [(InstantRain(meqn, msat), ForwardEuler(state))]
# InstantRain(meqn, msat)

# build time stepper
# stepper = PrescribedTransport(state,
#                               ((meqn, RK4(state)),))
stepper = PrescribedTransport(state,
                              ((meqn, ((SSPRK3(state), transport),)),),
                              physics_schemes=physics_schemes)

stepper.run(t=0, tmax=tmax)
