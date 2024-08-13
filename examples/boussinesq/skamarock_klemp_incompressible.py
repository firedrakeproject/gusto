"""
This example uses the incompressible Boussinesq equations to solve the vertical
slice gravity wave test case of Skamarock and Klemp, 1994:
``Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique'',
MWR.

Buoyancy is transported using SUPG, and the degree 1 elements are used.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from gusto import *
from firedrake import (as_vector, PeriodicIntervalMesh, ExtrudedMesh,
                       sin, SpatialCoordinate, Function, pi)
import sys

skamarock_klemp_incompressible_bouss_defaults = {
    'ncolumns': 100,
    'nlayers': 100,
    'dt': 1.0,
    'tmax': 600.,
    'dumpfreq': 200,
    'dirname': 'skamarock_klemp_incompressible_bouss'
}

def skamarock_klemp_incompressible_bouss(
        ncolumns=skamarock_klemp_incompressible_bouss_defaults['ncolumns'],
        nlayers=skamarock_klemp_incompressible_bouss_defaults['nlayers'],
        dt=skamarock_klemp_incompressible_bouss_defaults['dt'],
        tmax=skamarock_klemp_incompressible_bouss_defaults['tmax'],
        dumpfreq=skamarock_klemp_incompressible_bouss_defaults['dumpfreq'],
        dirname=skamarock_klemp_incompressible_bouss_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    dt = 6.
    L = 3.0e5  # Domain length
    H = 1.0e4  # Height position of the model top

    if '--running-tests' in sys.argv:
        tmax = dt
        dumpfreq = 1
        columns = 30  # number of columns
        nlayers = 5  # horizontal layers

    else:
        tmax = 3600.
        dumpfreq = int(tmax / (2*dt))
        columns = 300  # number of columns
        nlayers = 10  # horizontal layers

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, 'CG', 1)

    # Equation
    parameters = BoussinesqParameters()
    eqns = BoussinesqEquations(domain, parameters, compressible=False)

    # I/O
    output = OutputParameters(
        dirname='skamarock_klemp_incompressible',
        dumpfreq=dumpfreq,
        dumplist=['u'],
    )
    # list of diagnostic fields, each defined in a class in diagnostics.py
    diagnostic_fields = [CourantNumber(), Divergence(), Perturbation('b')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    b_opts = SUPGOptions()
    transported_fields = [TrapeziumRule(domain, "u"),
                        SSPRK3(domain, "b", options=b_opts)]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "b", ibp=b_opts.ibp)]

    # Linear solver
    linear_solver = BoussinesqSolver(eqns)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                    transport_methods,
                                    linear_solver=linear_solver)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    b0 = stepper.fields("b")
    p0 = stepper.fields("p")

    # spaces
    Vb = b0.function_space()

    x, z = SpatialCoordinate(mesh)

    # first setup the background buoyancy profile
    # z.grad(bref) = N**2
    N = parameters.N
    bref = z*(N**2)
    # interpolate the expression to the function
    b_b = Function(Vb).interpolate(bref)

    # setup constants
    a = 5.0e3
    deltab = 1.0e-2
    b_pert = deltab*sin(pi*z/H)/(1 + (x - L/2)**2/a**2)
    # interpolate the expression to the function
    b0.interpolate(b_b + b_pert)

    boussinesq_hydrostatic_balance(eqns, b_b, p0)

    uinit = (as_vector([20.0, 0.0]))
    u0.project(uinit)

    # set the background buoyancy
    stepper.set_reference_profiles([('b', b_b)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    # Run!
    stepper.run(t=0, tmax=tmax)

# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ncolumns',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=skamarock_klemp_incompressible_bouss_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=skamarock_klemp_incompressible_bouss_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=skamarock_klemp_incompressible_bouss_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=skamarock_klemp_incompressible_bouss_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=skamarock_klemp_incompressible_bouss_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=skamarock_klemp_incompressible_bouss_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    skamarock_klemp_incompressible_bouss(**vars(args))
