from gusto import *
from gusto import thermodynamics
from firedrake import (as_vector, SpatialCoordinate,
                       PeriodicRectangleMesh, ExtrudedMesh,
                       exp, cos, sin, cosh, sinh, tanh, pi, Function, sqrt)
import sys

day = 24.*60.*60.
hour = 60.*60.
dt = 30.
if '--running-tests' in sys.argv:
    tmax = dt
    tdump = dt
else:
    tmax = 30*day
    tdump = 2*hour

# set up mesh
columns = 30  # number of columns
L = 1000000.
m = PeriodicRectangleMesh(columns, 1, 2.*L, 1.e5, quadrilateral=True)

# build 2D mesh by extruding the base mesh
nlayers = 30  # horizontal layers
H = 10000.  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

# Coriolis expression
f = 1.e-04
Omega = as_vector([0., 0., f*0.5])

dirname = 'compressible_eady'
output = OutputParameters(dirname=dirname,
                          dumpfreq=int(tdump/dt),
                          dumplist=['u', 'rho', 'theta'],
                          perturbation_fields=['rho', 'theta', 'ExnerPi'],
                          log_level='INFO')

parameters = CompressibleEadyParameters(H=H, f=f)

diagnostic_fields = [CourantNumber(), VelocityY(),
                     ExnerPi(), ExnerPi(reference=True),
                     CompressibleKineticEnergy(),
                     CompressibleKineticEnergyY(),
                     CompressibleEadyPotentialEnergy(),
                     Sum("CompressibleKineticEnergy",
                         "CompressibleEadyPotentialEnergy"),
                     Difference("CompressibleKineticEnergy",
                                "CompressibleKineticEnergyY")]

state = State(mesh,
              dt=dt,
              output=output,
              parameters=parameters,
              diagnostic_fields=diagnostic_fields)

eqns = CompressibleEadyEquations(state, "RTCF", 1)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = state.spaces("HDiv")
Vt = state.spaces("theta")
Vr = state.spaces("DG")

# first setup the background buoyancy profile
# z.grad(bref) = N**2
# the following is symbolic algebra, using the default buoyancy frequency
# from the parameters class.
x, y, z = SpatialCoordinate(mesh)
g = parameters.g
Nsq = parameters.Nsq
theta_surf = parameters.theta_surf

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
theta_ref = theta_surf*exp(Nsq*(z-H/2)/g)
theta_b = Function(Vt).interpolate(theta_ref)


# set theta_pert
def coth(x):
    return cosh(x)/sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))


a = -4.5
Bu = 0.5
theta_exp = a*theta_surf/g*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)
                                      - n()*Bu*cosh(Z(z))*sin(pi*(x-L)/L))
theta_pert = Function(Vt).interpolate(theta_exp)

# set theta0
theta0.interpolate(theta_b + theta_pert)

# calculate hydrostatic Pi
rho_b = Function(Vr)
compressible_hydrostatic_balance(state, theta_b, rho_b)
compressible_hydrostatic_balance(state, theta0, rho0)

# set Pi0
Pi0 = calculate_Pi0(state, theta0, rho0)
state.parameters.Pi0 = Pi0

# set x component of velocity
cp = state.parameters.cp
dthetady = state.parameters.dthetady
Pi = thermodynamics.pi(state.parameters, rho0, theta0)
u = cp*dthetady/f*(Pi-Pi0)

# set y component of velocity
v = Function(Vr).assign(0.)
compressible_eady_initial_v(state, theta0, rho0, v)

# set initial u
u_exp = as_vector([u, v, 0.])
u0.project(u_exp)

# set the background profiles
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up transport schemes
transported_fields = [SSPRK3(state, "u"),
                   SSPRK3(state, "rho"),
                   SSPRK3(state, "theta")]

linear_solver = CompressibleSolver(state, eqns)

stepper = CrankNicolson(state, eqns, transported_fields,
                        linear_solver=linear_solver)

stepper.run(t=0, tmax=tmax)
