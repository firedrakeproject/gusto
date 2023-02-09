"""
The moist rising bubble test from Bryan & Fritsch (2002), in a cloudy
atmosphere.

The rise of the thermal is fueled by latent heating from condensation.
"""

from gusto import *
from gusto import thermodynamics
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt, exp,
                       NonlinearVariationalProblem, FunctionSpace,
                       NonlinearVariationalSolver, TestFunction, dx,
                       TrialFunction, Function, VectorFunctionSpace,
                       LinearVariationalProblem, LinearVariationalSolver,
                       errornorm, norm, plot)
import sys
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #
dt = 0.25
nt= 480
#tmaxs=[30, 30, 30, 30]
L = 120.0
deltax = 2.0
u_max = 1.0
cfl = 0.1

scheme = "Leapfrog"
#schemes = ["tr_bdf2"]

g1=(2.0 -sqrt(2.0))
# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

deltax=dt/cfl
#cfl = dts[i]/deltax
tmax = nt*dt
print('cfl',cfl)
print('dx',deltax)
# Domain
nx = int(L/deltax)
print('nx', nx)
mesh = PeriodicIntervalMesh(nx, L)
degree = 1
x =SpatialCoordinate(mesh)[0]
domain = Domain(mesh, dt, "CG", degree)

Vf = FunctionSpace(mesh, "CG", degree)
Vint = FunctionSpace(mesh, "CG", 3)
Vu = VectorFunctionSpace(mesh, "CG", degree+1)

# Equation
eqn = AdvectionEquation(domain, Vf, field_name="f", Vu=Vu)

# I/O
ndumps = 30
dirname = "%s_linear_transport_dt%s" % (scheme,dt)
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq)
io = IO(domain, output)
transport_scheme = Leapfrog(domain)
timestepper = PrescribedTransport(eqn, transport_scheme, io)

# define f0
xc = L / 2.0
ftemp= Function(Vint).interpolate(
    exp(-(x-xc)**2/20.0))
f0 =  Function(Vf).project(ftemp)

# Initial conditions
timestepper.fields("f").interpolate(f0)
timestepper.fields("u").project(as_vector([u_max]))
# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #
timestepper.run(t=0, tmax=tmax)
x_true=xc+dt*nt
if (x_true>L):
    x_true = x_true -L
print("x_true", x_true)
print("xc", xc)
true_temp= Function(Vint).interpolate(
    exp(-(x-x_true)**2/20.0))
true_sol =  Function(Vf).project(true_temp)
error_norm = errornorm(f0, timestepper.fields("f"),mesh=mesh)
norm_f = norm(f0, mesh=mesh)
print('error_norm', error_norm)
print('error', error_norm/norm_f)