from gusto import *
from firedrake import (as_vector, SpatialCoordinate,
                       PeriodicRectangleMesh, ExtrudedMesh,
                       cos, sin, cosh, sinh, tanh, pi, Function, sqrt)
import sys

day = 24.*60.*60.
hour = 60.*60.
dt = 100.
if '--running-tests' in sys.argv:
    tmax = dt
    tdump = dt
else:
    tmax = 30*day
    tdump = 2*hour

columns = 30
nlayers = 30
H = 10000.
L = 1000000.
f = 1.e-04

# rescaling
beta = 1.0
f = f/beta
L = beta*L

# Construct 2D periodic base mesh
m = PeriodicRectangleMesh(columns, 1, 2.*L, 1.e5, quadrilateral=True)

# build 3D mesh by extruding the base mesh
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)


output = OutputParameters(dirname='incompressible_eady',
                          dumpfreq=int(tdump/dt),
                          perturbation_fields=['p', 'b'],
                          log_level='INFO')

parameters = EadyParameters(H=H, L=L, f=f,
                            deltax=2.*L/float(columns),
                            deltaz=H/float(nlayers),
                            fourthorder=True)

diagnostic_fields = [CourantNumber(), VelocityY(),
                     KineticEnergy(), KineticEnergyY(),
                     EadyPotentialEnergy(),
                     Sum("KineticEnergy", "EadyPotentialEnergy"),
                     Difference("KineticEnergy", "KineticEnergyY"),
                     GeostrophicImbalance(), TrueResidualV()]

state = State(mesh,
              dt=dt,
              output=output,
              parameters=parameters,
              diagnostic_fields=diagnostic_fields)

# Coriolis expression
Omega = as_vector([0., 0., f*0.5])
eqns = IncompressibleEadyEquations(state, "RTCF", 1, Omega=Omega)

# Initial conditions
u0 = state.fields("u")
b0 = state.fields("b")
p0 = state.fields("p")

# spaces
Vu = u0.function_space()
Vb = b0.function_space()
Vp = p0.function_space()

# parameters
x, y, z = SpatialCoordinate(mesh)
Nsq = parameters.Nsq

# background buoyancy
bref = (z-H/2)*Nsq
b_b = Function(Vb).project(bref)


# buoyancy perturbation
def coth(x):
    return cosh(x)/sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))


a = -4.5
Bu = 0.5
b_exp = a*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)
                     - n()*Bu*cosh(Z(z))*sin(pi*(x-L)/L))
b_pert = Function(Vb).interpolate(b_exp)

# set total buoyancy
b0.project(b_b + b_pert)

# calculate hydrostatic pressure
p_b = Function(Vp)
incompressible_hydrostatic_balance(state, b_b, p_b)
incompressible_hydrostatic_balance(state, b0, p0)

# set x component of velocity
dbdy = parameters.dbdy
u = -dbdy/f*(z-H/2)

# set y component of velocity
v = Function(Vp).assign(0.)
eady_initial_v(state, p0, v)

# set initial u
u_exp = as_vector([u, v, 0.])
u0.project(u_exp)

# set the background profiles
state.set_reference_profiles([('p', p_b),
                              ('b', b_b)])

# Set up advection schemes
supg = True
if supg:
    b_opts = SUPGOptions()
else:
    b_opts = EmbeddedDGOptions()
advected_fields = [SSPRK3(state, "u"), SSPRK3(state, "b", options=b_opts)]

linear_solver = IncompressibleSolver(state, eqns)

stepper = CrankNicolson(state, eqns, advected_fields,
                        linear_solver=linear_solver)

stepper.run(t=0, tmax=tmax)
