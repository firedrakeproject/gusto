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
from pyop2.mpi import MPI
from firedrake import (
    PeriodicIntervalMesh, Function, assemble, SpatialCoordinate, COMM_WORLD,
    pi, sin, exp, dx
)
from gusto import (
    Domain, IO, OutputParameters, Timestepper, RK4, DGUpwind,
    ShallowWaterParameters, ShallowWaterEquations_1d, CGDiffusion,
    InteriorPenaltyDiffusion, DiffusionParameters, CoriolisOptions
)

shallow_water_1d_wave_defaults = {
    'ncells': 128,
    'dt': 0.0001,
    'tmax': 1.0,
    'dumpfreq': 1000,  # 10 outputs with default options
    'dirname': 'shallow_water_1d_wave'
}


def shallow_water_1d_wave(
        ncells=shallow_water_1d_wave_defaults['ncells'],
        dt=shallow_water_1d_wave_defaults['dt'],
        tmax=shallow_water_1d_wave_defaults['tmax'],
        dumpfreq=shallow_water_1d_wave_defaults['dumpfreq'],
        dirname=shallow_water_1d_wave_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    domain_length = 2*pi        # length of domain (m)
    kappa = 1.e-2               # diffusivity (m^2/s^2)
    epsilon = 0.1               # scaling factor for depth, gravity and rotation

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    mesh = PeriodicIntervalMesh(ncells, domain_length)
    domain = Domain(mesh, dt, 'CG', element_order)

    # Diffusion
    delta = domain_length / ncells
    u_diffusion_opts = DiffusionParameters(mesh, kappa=kappa)
    v_diffusion_opts = DiffusionParameters(mesh, kappa=kappa, mu=10/delta)
    D_diffusion_opts = DiffusionParameters(mesh, kappa=kappa, mu=10/delta)
    diffusion_options = [
        ("u", u_diffusion_opts),
        ("v", v_diffusion_opts),
        ("D", D_diffusion_opts)
    ]

    # Equation
    parameters = ShallowWaterParameters(mesh, rotation=CoriolisOptions.fplane,
                                        f0=1/epsilon, H=1/epsilon, g=1/epsilon)
    eqns = ShallowWaterEquations_1d(
        domain, parameters, diffusion_options=diffusion_options
    )

    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    io = IO(domain, output)

    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "v"),
        DGUpwind(eqns, "D")
    ]

    diffusion_methods = [
        CGDiffusion(eqns, "u", u_diffusion_opts),
        InteriorPenaltyDiffusion(eqns, "v", v_diffusion_opts),
        InteriorPenaltyDiffusion(eqns, "D", D_diffusion_opts)
    ]

    stepper = Timestepper(
        eqns, RK4(domain), io,
        spatial_methods=transport_methods+diffusion_methods
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    x = SpatialCoordinate(mesh)[0]
    D = stepper.fields("D")

    # Spatially-varying part of initial condition
    hexpr = (
        sin(x - pi/2) * exp(-4*(x - pi/2)**2)
        + sin(8*(x - pi)) * exp(-2*(x - pi)**2)
    )

    # Make a function to include spatially-varying part
    h = Function(D.function_space()).interpolate(hexpr)

    A = assemble(h*dx)

    # B must be the maximum value of h (across all ranks)
    B = np.zeros(1)
    COMM_WORLD.Allreduce(h.dat.data_ro.max(), B, MPI.MAX)

    # D is normalised form of h
    C0 = 1/(1 - 2*pi*B[0]/A)
    C1 = (1 - C0)/B[0]
    D.interpolate(parameters.H + C1*hexpr + C0)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(0, tmax)

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
