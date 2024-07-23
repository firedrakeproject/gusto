import numpy as np

from firedrake import *
from gusto import *
from pyop2.mpi import MPI

L = 2*pi
n = 128
delta = L/n
mesh = PeriodicIntervalMesh(128, L)
dt = 0.0001

domain = Domain(mesh, dt, 'CG', 1)

epsilon = 0.1
parameters = ShallowWaterParameters(H=1/epsilon, g=1/epsilon)

u_diffusion_opts = DiffusionParameters(kappa=1e-2)
v_diffusion_opts = DiffusionParameters(kappa=1e-2, mu=10/delta)
D_diffusion_opts = DiffusionParameters(kappa=1e-2, mu=10/delta)
diffusion_options = [("u", u_diffusion_opts),
                     ("v", v_diffusion_opts),
                     ("D", D_diffusion_opts)]

eqns = ShallowWaterEquations_1d(domain, parameters,
                                fexpr=Constant(1/epsilon),
                                diffusion_options=diffusion_options)

output = OutputParameters(dirname="1dsw_%s" % str(epsilon),
                          dumpfreq=50)
io = IO(domain, output)

transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "v"),
                     DGUpwind(eqns, "D")]

diffusion_methods = [CGDiffusion(eqns, "u", u_diffusion_opts),
                     InteriorPenaltyDiffusion(eqns, "v", v_diffusion_opts),
                     InteriorPenaltyDiffusion(eqns, "D", D_diffusion_opts)]

stepper = Timestepper(eqns, RK4(domain), io,
                      spatial_methods=transport_methods+diffusion_methods)

D = stepper.fields("D")
x = SpatialCoordinate(mesh)[0]
hexpr = (
    sin(x - pi/2) * exp(-4*(x - pi/2)**2)
    + sin(8*(x - pi)) * exp(-2*(x - pi)**2)
)
h = Function(D.function_space()).interpolate(hexpr)

A = assemble(h*dx)

# B must be the maximum value of h (across all ranks)
B = np.zeros(1)
COMM_WORLD.Allreduce(h.dat.data_ro.max(), B, MPI.MAX)

C0 = 1/(1 - 2*pi*B[0]/A)
C1 = (1 - C0)/B[0]
H = parameters.H
D.interpolate(C1*hexpr + C0)

D += parameters.H

stepper.run(0, 1)
