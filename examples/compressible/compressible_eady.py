"""
This solves the Eady problem using the compressible Euler equations.
"""

from gusto import *
from gusto import thermodynamics
from firedrake import (as_vector, SpatialCoordinate, solve, ds_b, ds_t,
                       PeriodicRectangleMesh, ExtrudedMesh,
                       exp, cos, sin, cosh, sinh, tanh, pi, Function, sqrt)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.
hour = 60.*60.
dt = 30.
L = 1000000.
H = 10000.  # Height position of the model top
f = 1.e-04

if '--running-tests' in sys.argv:
    tmax = dt
    tdump = dt
    columns = 10  # number of columns
    nlayers = 5  # horizontal layers
else:
    tmax = 30*day
    tdump = 5*day
    columns = 30  # number of columns
    nlayers = 30  # horizontal layers

dirname = 'compressible_eady'

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain -- 2D periodic base mesh which is one cell thick
m = PeriodicRectangleMesh(columns, 1, 2.*L, 1.e5, quadrilateral=True)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, "RTCF", 1)

# Equation
Omega = as_vector([0., 0., f*0.5])
parameters = CompressibleEadyParameters(H=H, f=f)
eqns = CompressibleEadyEquations(domain, parameters, Omega=Omega)

# I/O
output = OutputParameters(dirname=dirname,
                          dumpfreq=int(tdump/dt))

diagnostic_fields = [CourantNumber(), YComponent('u'),
                     Exner(parameters), Exner(parameters, reference=True),
                     CompressibleKineticEnergy(),
                     CompressibleKineticEnergyY(),
                     CompressibleEadyPotentialEnergy(parameters),
                     Sum("CompressibleKineticEnergy",
                         "CompressibleEadyPotentialEnergy"),
                     Difference("CompressibleKineticEnergy",
                                "CompressibleKineticEnergyY"),
                     Perturbation('rho'), Perturbation('theta'),
                     Perturbation('Exner')]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes and methods
theta_opts = SUPGOptions()
transport_schemes = [SSPRK3(domain, "u"),
                     SSPRK3(domain, "rho"),
                     SSPRK3(domain, "theta", options=theta_opts)]
transport_methods = [DGUpwind(eqns, "u"),
                     DGUpwind(eqns, "rho"),
                     DGUpwind(eqns, "theta", ibp=theta_opts.ibp)]

# Linear solver
linear_solver = CompressibleSolver(eqns)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transport_schemes,
                                  transport_methods,
                                  linear_solver=linear_solver)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# spaces
Vu = domain.spaces("HDiv")
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")

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
compressible_hydrostatic_balance(eqns, theta_b, rho_b, solve_for_rho=True)
compressible_hydrostatic_balance(eqns, theta0, rho0, solve_for_rho=True)

# set Pi0 -- have to get this from the equations
Pi0 = eqns.prescribed_fields('Pi0')
Pi0.interpolate(exner_pressure(parameters, rho0, theta0))

# set x component of velocity
cp = parameters.cp
dthetady = parameters.dthetady
Pi = thermodynamics.exner_pressure(parameters, rho0, theta0)
u = cp*dthetady/f*(Pi-Pi0)

# set y component of velocity by solving a problem
v = Function(Vr).assign(0.)

# get Pi gradient
g = TrialFunction(Vu)
wg = TestFunction(Vu)

n = FacetNormal(mesh)

a = inner(wg, g)*dx
L = -div(wg)*Pi*dx + inner(wg, n)*Pi*(ds_t + ds_b)
pgrad = Function(Vu)
solve(a == L, pgrad)

# get initial v
m = TrialFunction(Vr)
phi = TestFunction(Vr)

a = phi*f*m*dx
L = phi*cp*theta0*pgrad[0]*dx
solve(a == L, v)

# set initial u
u_exp = as_vector([u, v, 0.])
u0.project(u_exp)

# set the background profiles
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
