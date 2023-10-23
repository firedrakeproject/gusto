"""
The Eady problem solved using the incompressible Boussinesq equations.
"""

from gusto import *
from firedrake import (as_vector, SpatialCoordinate, solve, ds_t, ds_b,
                       PeriodicRectangleMesh, ExtrudedMesh,
                       cos, sin, cosh, sinh, tanh, pi, Function, sqrt)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.
hour = 60.*60.
dt = 100.

if '--running-tests' in sys.argv:
    tmax = dt
    tdump = dt
    columns = 10
    nlayers = 5
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

dirname = 'incompressible_eady'

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain -- 2D periodic base mesh which is one cell thick
m = PeriodicRectangleMesh(columns, 1, 2.*L, 1.e5, quadrilateral=True)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, "RTCF", 1)

# Equation
Omega = as_vector([0., 0., f*0.5])
parameters = EadyParameters(H=H, L=L, f=f,
                            deltax=2.*L/float(columns),
                            deltaz=H/float(nlayers),
                            fourthorder=True)
eqns = IncompressibleEadyEquations(domain, parameters, Omega=Omega)

# I/O
output = OutputParameters(dirname=dirname,
                          dumpfreq=int(tdump/dt))

diagnostic_fields = [CourantNumber(), YComponent('u'),
                     KineticEnergy(), KineticEnergyY(),
                     IncompressibleEadyPotentialEnergy(parameters),
                     Sum("KineticEnergy", "EadyPotentialEnergy"),
                     Difference("KineticEnergy", "KineticEnergyY"),
                     IncompressibleGeostrophicImbalance(eqns),
                     TrueResidualV(parameters), SawyerEliassenU(eqns),
                     Perturbation('p'), Perturbation('b')]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes and methods
b_opts = SUPGOptions()
transport_schemes = [SSPRK3(domain, "u"), SSPRK3(domain, "b", options=b_opts)]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "b", ibp=b_opts.ibp)]

# Linear solve
linear_solver = IncompressibleSolver(eqns)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transport_schemes,
                                  transport_methods,
                                  linear_solver=linear_solver)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

# Initial conditions
u0 = stepper.fields("u")
b0 = stepper.fields("b")
p0 = stepper.fields("p")

# spaces
Vu = domain.spaces("HDiv")
Vb = domain.spaces("theta")
Vp = domain.spaces("DG")

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
incompressible_hydrostatic_balance(eqns, b_b, p_b)
incompressible_hydrostatic_balance(eqns, b0, p0)

# set x component of velocity
dbdy = parameters.dbdy
u = -dbdy/f*(z-H/2)

# set y component of velocity
v = Function(Vp).assign(0.)

g = TrialFunction(Vu)
wg = TestFunction(Vu)

n = FacetNormal(mesh)

a = inner(wg, g)*dx
L = -div(wg)*p0*dx + inner(wg, n)*p0*(ds_t + ds_b)
pgrad = Function(Vu)
solve(a == L, pgrad)

# get initial v
Vp = p0.function_space()
phi = TestFunction(Vp)
m = TrialFunction(Vp)

a = f*phi*m*dx
L = phi*pgrad[0]*dx
solve(a == L, v)

# set initial u
u_exp = as_vector([u, v, 0.])
u0.project(u_exp)

# set the background profiles
stepper.set_reference_profiles([('p', p_b),
                                ('b', b_b)])

# The residual diagnostic needs to have u_n added to stepper.fields
u_n = stepper.x.n('u')
stepper.fields('u_n', field=u_n)

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
