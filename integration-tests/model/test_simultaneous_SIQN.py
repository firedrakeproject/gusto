"""
This tests the use of simultaneous transport with the SIQN
timestepping method. A few timesteps are taken with the Bryan-Fritsch
bubble test case, which solves the Compressible Euler Equations.
The two tracers of water vapour and cloud water are being tranpsported
conservatively, which means they need to be transported simultaneously
with the density.
Degree 0 and 1 configurations are  tested to ensure that the simultaneous
transport is working with the different wrappers.
"""

from os.path import join, abspath, dirname
from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, conditional, cos, pi,
    sqrt, NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction,
    dx, TrialFunction, Function, as_vector, LinearVariationalProblem,
    LinearVariationalSolver, Constant, BrokenElement
)
from gusto import *
import pytest


def run_simult_SIQN(tmpdir, order):

    if order == 0:
        ncolumns = 20
        nlayers = 20
        u_eqn_type = "vector_advection_form"
    else:
        ncolumns = 10
        nlayers = 10
        u_eqn_type = "vector_invariant_form"

    dt = 2.0
    tmax = 10.0

    domain_width = 10000.     # domain width, in m
    domain_height = 10000.    # domain height, in m
    zc = 2000.                # vertical centre of bubble, in m
    rc = 2000.                # radius of bubble, in m
    Tdash = 2.0               # strength of temperature perturbation, in K
    Tsurf = 320.0             # background theta_e value, in K
    total_water = 0.02        # total moisture mixing ratio, in kg/kg

    # Domain
    mesh_name = 'bryan_fritsch_mesh'
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(
        base_mesh, layers=nlayers, layer_height=domain_height/nlayers, name=mesh_name
    )
    domain = Domain(mesh, dt, 'CG', order)

    # Set up the tracers and their transport schemes
    V_rho = domain.spaces('DG')
    V_theta = domain.spaces('theta')

    tracers = [WaterVapour(space='theta',
                           transport_eqn=TransportEquationType.tracer_conservative,
                           density_name='rho'),
               CloudWater(space='theta',
                          transport_eqn=TransportEquationType.tracer_conservative,
                          density_name='rho')]

    # Equation
    params = CompressibleParameters(mesh)
    eqns = CompressibleEulerEquations(
        domain, params, active_tracers=tracers, u_transport_option=u_eqn_type
    )

    # I/O
    output_dirname = tmpdir+"/simult_SIQN_order"+str(order)

    output = OutputParameters(
        dirname=output_dirname, dumpfreq=5, chkptfreq=5, checkpoint=True
    )
    io = IO(domain, output)

    # Set up transport schemes
    if order == 0:
        VDG1 = domain.spaces("DG1_equispaced")
        VCG1 = FunctionSpace(mesh, "CG", 1)
        Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element())
        Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

        u_opts = RecoveryOptions(embedding_space=Vu_DG1,
                                 recovered_space=Vu_CG1,
                                 boundary_method=BoundaryMethod.taylor)
        theta_opts = RecoveryOptions(embedding_space=VDG1,
                                     recovered_space=VCG1)

        suboptions = {'rho': RecoveryOptions(embedding_space=VDG1,
                                             recovered_space=VCG1,
                                             boundary_method=BoundaryMethod.taylor),
                      'water_vapour': ConservativeRecoveryOptions(embedding_space=VDG1,
                                                                  recovered_space=VCG1,
                                                                  rho_name="rho",
                                                                  orig_rho_space=V_rho),
                      'cloud_water': ConservativeRecoveryOptions(embedding_space=VDG1,
                                                                 recovered_space=VCG1,
                                                                 rho_name="rho",
                                                                 orig_rho_space=V_rho)}
    else:
        theta_opts = EmbeddedDGOptions()
        Vt_brok = FunctionSpace(mesh, BrokenElement(V_theta.ufl_element()))
        suboptions = {'rho': EmbeddedDGOptions(embedding_space=Vt_brok),
                      'water_vapour': ConservativeEmbeddedDGOptions(embedding_space=Vt_brok,
                                                                    rho_name="rho",
                                                                    orig_rho_space=V_rho),
                      'cloud_water': ConservativeEmbeddedDGOptions(embedding_space=Vt_brok,
                                                                   rho_name="rho",
                                                                   orig_rho_space=V_rho)}

    transported_fields = [SSPRK3(domain, "theta", options=theta_opts)]

    mixed_opts = MixedFSOptions(suboptions=suboptions)
    transported_fields.append(SSPRK3(domain, ["rho", "water_vapour", "cloud_water"], options=mixed_opts, rk_formulation=RungeKuttaFormulation.predictor))

    if order == 0:
        transported_fields.append(SSPRK3(domain, 'u', options=u_opts))
    else:
        transported_fields.append(TrapeziumRule(domain, 'u'))

    transport_methods = [
        DGUpwind(eqns, field) for field in
        ["u", "rho", "theta", "water_vapour", "cloud_water"]
    ]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Physics schemes (condensation/evaporation)
    physics_schemes = [(SaturationAdjustment(eqns), ForwardEuler(domain))]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver, final_physics_schemes=physics_schemes
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
    x, z = SpatialCoordinate(mesh)
    quadrature_degree = (4, 4)
    dxp = dx(degree=(quadrature_degree))

    # Define constant theta_e and water_t
    theta_e = Function(Vt).assign(Tsurf)
    water_t = Function(Vt).assign(total_water)

    # Calculate hydrostatic fields
    saturated_hydrostatic_balance(eqns, stepper.fields, theta_e, water_t)

    # make mean fields
    theta_b = Function(Vt).assign(theta0)
    rho_b = Function(Vr).assign(rho0)
    water_vb = Function(Vt).assign(water_v0)
    water_cb = Function(Vt).assign(water_t - water_vb)

    # define perturbation
    xc = domain_width / 2
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    theta_pert = Function(Vt).interpolate(
        conditional(
            r > rc,
            0.0,
            Tdash * (cos(pi * r / (2.0 * rc))) ** 2
        )
    )

    # define initial theta
    theta0.interpolate(theta_b * (theta_pert / 300.0 + 1.0))

    # find perturbed rho
    gamma = TestFunction(Vr)
    rho_trial = TrialFunction(Vr)
    a = gamma * rho_trial * dxp
    L = gamma * (rho_b * theta_b / theta0) * dxp
    rho_problem = LinearVariationalProblem(a, L, rho0)
    rho_solver = LinearVariationalSolver(rho_problem)
    rho_solver.solve()

    # find perturbed water_v
    w_v = Function(Vt)
    phi = TestFunction(Vt)
    rho_averaged = Function(Vt)
    rho_recoverer = Recoverer(rho0, rho_averaged)
    rho_recoverer.project()

    exner = thermodynamics.exner_pressure(eqns.parameters, rho_averaged, theta0)
    p = thermodynamics.p(eqns.parameters, exner)
    T = thermodynamics.T(eqns.parameters, theta0, exner, r_v=w_v)
    w_sat = thermodynamics.r_sat(eqns.parameters, T, p)

    w_functional = (phi * w_v * dxp - phi * w_sat * dxp)
    w_problem = NonlinearVariationalProblem(w_functional, w_v)
    w_solver = NonlinearVariationalSolver(w_problem)
    w_solver.solve()

    water_v0.assign(w_v)
    water_c0.assign(water_t - water_v0)

    # wind initially zero
    u0.project(as_vector(
        [Constant(0.0), Constant(0.0)]
    ))

    stepper.set_reference_profiles(
        [
            ('rho', rho_b),
            ('theta', theta_b),
            ('water_vapour', water_vb),
            ('cloud_water', water_cb)
        ]
    )

    # --------------------------------------------------------------------- #
    # Run
    # --------------------------------------------------------------------- #

    stepper.run(t=0, tmax=tmax)

    # State for checking checkpoints
    checkpoint_name = 'simult_SIQN_order'+str(order)+'_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=output_dirname,
                                    checkpoint_pickup_filename=new_path,
                                    checkpoint=True)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, "CG", order)
    check_params = CompressibleParameters(check_mesh)
    check_eqn = CompressibleEulerEquations(check_domain, check_params, active_tracers=tracers, u_transport_option=u_eqn_type)
    check_io = IO(check_domain, check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [], [])
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)

    return stepper, check_stepper


@pytest.mark.parametrize("order", [0, 1])
def test_simult_SIQN(tmpdir, order):

    dirname = str(tmpdir)
    stepper, check_stepper = run_simult_SIQN(dirname, order)

    for variable in ['u', 'rho', 'theta', 'water_vapour', 'cloud_water']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        diff_array = new_variable.dat.data - check_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(check_variable.dat.data)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in the ' + \
            f'order {order} elements test do not match KGO values'
