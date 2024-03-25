from firedrake import *
from gusto import *
import matplotlib.pyplot as plt

L = 2*pi
n = 128
delta = L/n
mesh = PeriodicIntervalMesh(128, 2*pi)
dt = 0.0001

domain = Domain(mesh, dt, 'CG', 1)

for epsilon in [0.1]:
    parameters = ShallowWaterParameters(H=1/epsilon, g=1/epsilon)
    diffusion_options = [("u", DiffusionParameters(kappa=1e-2, mu=10/delta)),
                         ("v", DiffusionParameters(kappa=1e-2, mu=10/delta)),
                         ("D", DiffusionParameters(kappa=1e-2, mu=10/delta))]
    eqns = ShallowWaterEquations_1d(domain, parameters,
                                    fexpr=Constant(1/epsilon),
                                    diffusion_options=diffusion_options)
    output = OutputParameters(dirname="1dsw_%s" % str(epsilon),
                              dumpfreq=50,
                              log_courant=False)
    io = IO(domain, output)
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "v"),
                         DGUpwind(eqns, "D")]
    #stepper = Timestepper(eqns, ImplicitMidpoint(domain), io)
    stepper = Timestepper(eqns, RK4(domain), io,
                          spatial_methods=transport_methods)

    D = stepper.fields("D")
    x = SpatialCoordinate(mesh)[0]
    hexpr = (
        exp(-4*(x-0.5*pi)**2) * sin((x-0.5*pi)) +
        exp(-2*(x-1*pi)**2) * sin(8*(x-1*pi))
    )
    h = Function(D.function_space()).interpolate(hexpr)
    #plot(h)
    #plt.show()

    A = assemble(h*dx)
    B = h.dat.data.max()
    C0 = 1/(1-2*pi*B/A)
    C1 = (1-C0)/B
    H = parameters.H
    D.interpolate(C1*hexpr + C0)
    #plot(D)
    #plt.show()
    print(assemble(D*dx))
    print(D.dat.data.max())
    #D += parameters.H

    #D.interpolate(H + 0.1*H*exp(-(x-pi)**2/0.5))

    stepper.run(0, 10)
