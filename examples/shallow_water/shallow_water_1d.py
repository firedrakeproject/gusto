"""
A shallow water wave on a 1D periodic domain. The test is taken from
Haut & Wingate, 2014:
``An asymptotic parallel-in-time method for highly oscillatory PDEs'', SIAM JSC.

The velocity includes a component normal to the domain, and diffusion terms are
included in the equations.

This example uses an explicit RK4 timestepper to solve the equations.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import sys

from firedrake import *
from gusto import *
from pyop2.mpi import MPI

shallow_water_1d_wave_defaults = {
    'ncells': 48,
    'dt': 300.0,               # 5 minutes
    'tmax': 6.*24.*60.*60.,    # 6 days
    'dumpfreq': 288,           # once per day with default options
    'dirname': 'shallow_water_1d_wave'
}

def shallow_water_1d_wave(
        ncells=shallow_water_1d_wave_defaults['ncells'],
        dt=shallow_water_1d_wave_defaults['dt'],
        tmax=shallow_water_1d_wave_defaults['tmax'],
        dumpfreq=shallow_water_1d_wave_defaults['dumpfreq'],
        dirname=shallow_water_1d_wave_defaults['dirname']
):

    L = 2*pi
    n = 128
    delta = L/n
    mesh = PeriodicIntervalMesh(128, L)
    dt = 0.0001
    if '--running-tests' in sys.argv:
        T = 0.0005
    else:
        T = 1

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

    stepper.run(0, T)

# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ncells',
        help="The number of cells in the 1D domain",
        type=int,
        default=shallow_water_1d_wave_defaults['ncells']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=shallow_water_1d_wave_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=shallow_water_1d_wave_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=shallow_water_1d_wave_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=shallow_water_1d_wave_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    shallow_water_1d_wave(**vars(args))
