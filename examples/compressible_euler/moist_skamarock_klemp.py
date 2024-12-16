"""
A moist version of the non-hydrostatic vertical slice gravity wave test case of
Skamarock and Klemp, as described in Bendall et al, 2020:
``A compatible finite‐element discretisation for the moist compressible Euler
equations'', QJRMS.

The gravity wave propagates through a saturated and cloudy atmosphere, so that
evaporation and condensation impact the development of the wave.

This test uses some different options to other tests:
- wind is transported in a mixed velocity-vorticity form
- density is transported with an advective-then-flux formulation
- moisture variables are transported conservatively
- the reference profiles are the start-of-time step states
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
import itertools
from firedrake import (
    as_vector, SpatialCoordinate, PeriodicIntervalMesh, ExtrudedMesh, exp, sin,
    Function, pi, errornorm, TestFunction, dx, BrokenElement, FunctionSpace,
    NonlinearVariationalProblem, NonlinearVariationalSolver
)
import numpy as np
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    VorticityTransport, Perturbation, thermodynamics, CompressibleParameters,
    CompressibleEulerEquations, HydrosaticCompressibleEulerEquations,
    compressible_hydrostatic_balance, RungeKuttaFormulation, CompressibleSolver,
    WaterVapour, CloudWater, Theta_e, Recoverer, saturated_hydrostatic_balance,
)

moist_skamarock_klemp_defaults = {
    'ncolumns': 150,
    'nlayers': 10,
    'dt': 6.0,
    'tmax': 3000.,
    'dumpfreq': 250,
    'dirname': 'moist_skamarock_klemp'
}


def moist_skamarock_klemp(
        ncolumns=moist_skamarock_klemp_defaults['ncolumns'],
        nlayers=moist_skamarock_klemp_defaults['nlayers'],
        dt=moist_skamarock_klemp_defaults['dt'],
        tmax=moist_skamarock_klemp_defaults['tmax'],
        dumpfreq=moist_skamarock_klemp_defaults['dumpfreq'],
        dirname=moist_skamarock_klemp_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    domain_width = 3.0e5      # Width of domain (m)
    domain_height = 1.0e4     # Height of domain (m)
    Tsurf = 300.              # Temperature at surface (K)
    wind_initial = 20.        # Initial wind in x direction (m/s)
    pert_width = 5.0e3        # Width parameter of perturbation (m)
    deltaTheta = 1.0e-2       # Magnitude of theta perturbation (K)
    N = 0.01                  # Brunt-Vaisala frequency (1/s)
    total_water = 0.02        # total moisture mixing ratio, in kg/kg

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1
    alpha = 0.5
    u_eqn_type = 'vector_invariant_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    domain = Domain(mesh, dt, "CG", element_order)

    # Equation
    parameters = CompressibleParameters()
    tracers = [WaterVapour(), CloudWater()]
    eqns = CompressibleEulerEquations(
        domain, parameters, active_tracers=tracers,
        u_transport_option=u_eqn_type
    )

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True
    )

    diagnostic_fields = [Theta_e(eqns), Perturbation('Theta_e')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    vorticity_form = VorticityTransport(
        domain, domain.spaces("HDiv"), domain.spaces("H1"), supg=True
    )
    Vt_brok = FunctionSpace(mesh, BrokenElement(V_theta.ufl_element()))
    embedded_dg = EmbeddedDGOptions(embedding_space=Vt_brok)

    transported_fields = [
        SSPRK3(
            domain, "u", subcycle_by_courant=0.25, augmentation=vorticity_form,
            rk_formulation=RungeKuttaFormulation.predictor
        ),
        SSPRK3(
            domain, "rho", subcycle_by_courant=0.25,
            rk_formulation=RungeKuttaFormulation.linear
        ),
        SSPRK3(domain, "theta", subcycle_by_courant=0.25, opts=embedded_dg),
        SSPRK3(
            domain, "water_vapour", subcycle_by_courant=0.25, opts=embedded_dg
        ),
        SSPRK3(
            domain, "cloud_water", subcycle_by_courant=0.25, opts=embedded_dg
        )
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho", advective_then_flux=True),
        DGUpwind(eqns, "theta"),
        DGUpwind(eqns, "water_vapour"),
        DGUpwind(eqns, "cloud_water")
    ]

    # Linear solver
    tau_values = {'rho': 1.0, 'theta': 1.0}
    linear_solver = CompressibleSolver(eqns, alpha=alpha, tau_values=tau_values)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver, alpha=alpha, reference_update_freq=1,
        accelerator=True
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")
    water_v0 = stepper.fields("water_vapour")
    water_c0 = stepper.fields("cloud_water")

    # spaces
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = parameters.g

    x, z = SpatialCoordinate(mesh)
    quadrature_degree = (4, 4)
    dxp = dx(degree=(quadrature_degree))

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    theta_e = Function(Vt).interpolate(Tsurf*exp(N**2*z/g))
    water_t = Function(Vt).assign(total_water)

    # Calculate hydrostatic fields
    saturated_hydrostatic_balance(eqns, stepper.fields, theta_e, water_t)

    # Add perturbation to theta_e ----------------------------------------------
    theta_e_pert = (
        deltaTheta * sin(pi*z/domain_height)
        / (1 + (x - domain_width/2)**2 / pert_width**2)
    )
    theta_e.interpolate(theta_e + theta_e_pert)

    # Find perturbed water_v and theta_vd --------------------------------------
    # expressions for finding theta0 and water_v0 from theta_e and water_t
    rho_averaged = Function(Vt)
    rho_recoverer = Recoverer(rho0, rho_averaged)

    w_h = Function(Vt)
    theta_h = Function(Vt)
    theta_e_test = Function(Vt)
    pie = thermodynamics.pi(state.parameters, rho_averaged, theta0)
    p = thermodynamics.p(state.parameters, pie)
    T = thermodynamics.T(state.parameters, theta0, pie, water_v0)
    r_v_expr = thermodynamics.r_sat(state.parameters, T, p)
    theta_e_expr = thermodynamics.theta_e(state.parameters, T, p, water_v0, water_t)
    rho_h = Function(Vr)
    zeta = TestFunction(Vr)

    rho_form = zeta * rho_h * theta0 * dxp - zeta * rho_b * theta_b * dxp
    rho_prob = NonlinearVariationalProblem(rho_form, rho_h)
    rho_solver = NonlinearVariationalSolver(rho_prob)

    max_outer_solve_count = 40
    max_theta_solve_count = 15
    max_inner_solve_count = 5
    delta = 0.8

    for i in range(max_outer_solve_count):

        rho_solver.solve()
        rho0.assign(rho0 * (1 - delta) + delta * rho_h)
        rho_recoverer.project()

        theta_e_test.assign(theta_e_expr)
        if errornorm(theta_e_test, theta_e) < 1e-6:
            break

        for j in range(max_theta_solve_count):
            theta_h.interpolate(theta_e / theta_e_expr * theta0)
            theta0.assign(theta0 * (1 - delta) + delta * theta_h)

            # break when close enough
            if errornorm(theta_e_test, theta_e) < 1e-6:
                break
            for k in range(max_inner_solve_count):
                w_h.interpolate(r_v_expr)
                water_v0.assign(water_v0 * (1 - delta) + delta * w_h)

                # break when close enough
                theta_e_test.assign(theta_e_expr)
                if errornorm(theta_e_test, theta_e) < 1e-6:
                    break

    water_c0.assign(water_t - water_v0)

    theta_e.interpolate(theta_e + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([wind_initial, 0.0]))

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
        '--ncolumns',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=moist_skamarock_klemp_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=moist_skamarock_klemp_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=moist_skamarock_klemp_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=moist_skamarock_klemp_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=moist_skamarock_klemp_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=moist_skamarock_klemp_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    moist_skamarock_klemp(**vars(args))