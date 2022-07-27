from gusto import *
from firedrake import (PeriodicIntervalMesh, SpatialCoordinate, FunctionSpace,
                       VectorFunctionSpace, conditional, cos, pi)

# set up mesh
Lx = 10000
delta = 50
nx = int(Lx/delta)

mesh = PeriodicIntervalMesh(nx, Lx)
x = SpatialCoordinate(mesh)[0]

dirname = 'forced_advection'

output = OutputParameters(dirname=dirname, dumpfreq=1)

diagnostic_fields = [CourantNumber()]

# set up parameters
dt = 15
u_max = 1
qmax = 7
qh = 2
Csat = 7.5
Ksat = 2.5
x1 = Lx/4
x2 = (3*Lx)/8

state = State(mesh,
              dt=dt,
              output=output,
              parameters=None,
              diagnostics=None,
              diagnostic_fields=diagnostic_fields)

# set up function spaces
VD = FunctionSpace(mesh, "DG", 1)
Vu = VectorFunctionSpace(mesh, "CG", 1)

# initial moisture profile
mexpr = conditional(x < (3*Lx)/8, conditional(x > Lx/4, qmax, qmax-qh), qmax-qh)

# define saturation profile
# coordinate = (x-(3*Lx)/2)/Lx
msat_expr = Csat + (Ksat * cos(2*pi*(x/Lx)))
msat = Function(VD)
msat.interpolate(msat_expr)

# set up advection equation
meqn = AdvectionEquation(state, VD, field_name="m", Vu=Vu)
state.fields("u").project(as_vector([u_max]))
state.fields("m").project(mexpr)

# define rain variable
rain = state.fields("r", VD)
rain.project(Constant(0.5))

# set up moisture as a field to be transported
transported_fields = [SSPRK3(state, "m"), ImplicitMidpoint(state, "u")]

# add instant rain forcing
physics_list = [InstantRain(state, msat)]


# prescribe velocity for transport
def transport_u(t):
    return state.fields("u")


# build time stepper
stepper = PrescribedTransport(state, ((meqn, SSPRK3(state)),),
                              physics_list=physics_list,
                              prescribed_transporting_velocity=transport_u)
stepper.run(t=0, tmax=500*dt)
