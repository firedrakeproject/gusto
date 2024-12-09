"""
Test Case 2 (solid-body rotation with geostrophically-balanced flow) of
Williamson et al, 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

The example here uses the icosahedral sphere mesh and degree 1 spaces.
"""

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import SpatialCoordinate, sin, cos, pi, Function, exp, IcosahedralSphereMesh, interpolate
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, ShallowWaterParameters, ShallowWaterEquations,
    RelativeVorticity, PotentialVorticity, SteadyStateError,
    ShallowWaterKineticEnergy, ShallowWaterPotentialEnergy,
    ShallowWaterPotentialEnstrophy, rotated_lonlatr_coords,
    ZonalComponent, MeridionalComponent, rotated_lonlatr_vectors,
    GeneralIcosahedralSphereMesh, AdvectionEquation, logger, VectorFunctionSpace,
    lonlatr_from_xyz
)

dir = '/data/home/sh1293/results/passive_tracer_williamson_2_tracer-tophat_longer'
days = 300

def initial_T(X, rlat, Tini):
    lats = []
    for X0 in X:
        x, y, z = X0
        _, lat, _ = lonlatr_from_xyz(x, y, z)
        lats.append(lat)
    return np.interp(np.array(lats), rlat, Tini)



williamson_2_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': days*24.*60.*60.,    # 5 days
    'dumpfreq': 96,            # once per day with default options
    'dirname': 'williamson_2'
}


def williamson_2(
        ncells_per_edge=williamson_2_defaults['ncells_per_edge'],
        dt=williamson_2_defaults['dt'],
        tmax=williamson_2_defaults['tmax'],
        dumpfreq=williamson_2_defaults['dumpfreq'],
        dirname=williamson_2_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.                  # planetary radius (m)
    mean_depth = 5960.                 # reference depth (m)
    rotate_pole_to = (0.0, pi/3)       # location of North pole of mesh
    u_max = 2*pi*radius/(12*24*60*60)  # Max amplitude of the zonal wind (m/s)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1
    u_eqn_type = 'vector_invariant_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = IcosahedralSphereMesh(radius, refinement_level=4, degree=1)
    domain = Domain(mesh, dt, 'BDM', element_order, rotated_pole=rotate_pole_to)
    xyz = SpatialCoordinate(mesh)

    # Equation
    parameters = ShallowWaterParameters(H=mean_depth)
    Omega = parameters.Omega
    lon, lat, r = rotated_lonlatr_coords(xyz, rotate_pole_to)
    e_lon, _, _ = rotated_lonlatr_vectors(xyz, rotate_pole_to)
    fexpr = 2*Omega*sin(lat)
    eqns = ShallowWaterEquations(
        domain, parameters, fexpr=fexpr, u_transport_option=u_eqn_type)
    tracer_eqn = AdvectionEquation(domain, domain.spaces("DG"), "tracer")

    logger.info(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} \n mpiexec -n nprocs python script.py')

    # I/O
    output = OutputParameters(
        dirname=dir, dumpfreq=dumpfreq, dump_nc=True,
        dumplist_latlon=['D', 'D_error', 'tracer'],
    )
    diagnostic_fields = [
        RelativeVorticity(), SteadyStateError('RelativeVorticity'),
        PotentialVorticity(), ShallowWaterKineticEnergy(),
        ShallowWaterPotentialEnergy(parameters),
        ShallowWaterPotentialEnstrophy(),
        SteadyStateError('u'), SteadyStateError('D'),
        MeridionalComponent('u', rotate_pole_to),
        ZonalComponent('u', rotate_pole_to)
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "D", fixed_subcycles=2)]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "D"),
        DGUpwind(tracer_eqn, "tracer")
    ]
    tracer_transport = [(tracer_eqn, SSPRK3(domain))]


    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods, auxiliary_equations_and_schemes=tracer_transport
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    g = parameters.g

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    tracer0 = stepper.fields("tracer")

    rlat = np.linspace(-np.pi/2, np.pi/2, num=1000000)[1:-1]
    Tini = np.where(rlat>=80*pi/180, 1, 0)

    VT = tracer0.function_space()
    Tmesh = VT.mesh()
    WT = VectorFunctionSpace(Tmesh, VT.ufl_element())
    XT = interpolate(Tmesh.coordinates, WT)
    tracer0.dat.data[:] = initial_T(XT.dat.data_ro, rlat, Tini)

    uexpr = u_max*cos(lat)*e_lon
    Dexpr = mean_depth - (radius * Omega * u_max + 0.5*u_max**2)*(sin(lat))**2/g
    x = SpatialCoordinate(mesh)
    # f_init = exp(-x[2]**2 - x[0]**2)
    # f_init = 5*exp(-x[0]**2)
    f_init = exp(-(x[2]/1e6)**2-(x[0]/1e6)**2)
    # f_init = 1

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    # tracer0.interpolate(f_init)

    Dbar = Function(D0.function_space()).assign(mean_depth)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

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
        '--ncells_per_edge',
        help="The number of cells per edge of icosahedron",
        type=int,
        default=williamson_2_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=williamson_2_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=williamson_2_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=williamson_2_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=williamson_2_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    williamson_2(**vars(args))

    print(f'{dir}')