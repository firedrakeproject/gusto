"""
Tests the split_timestepper.

First test: checks that the splitting of 
advection and transport terms is performed correctly.

Second test: checks that the splitting of
physics and individual terms in the RSWEs
is performed correctly.
"""

from os.path import join, abspath, dirname
from netCDF4 import Dataset
from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp, as_vector,
                       norm, Constant, conditional, sqrt, VectorFunctionSpace,
                       pi, IcosahedralSphereMesh, acos, sin, cos, max_value, 
                       min_value, errornorm, ExtrudedMesh)
from gusto import *
import gusto.equations.thermodynamics as tde
import pytest


def run_split_timestepper_adv_diff(tmpdir, timestepper):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    dt = 0.02
    tmax = 1.0
    L = 10
    mesh = PeriodicIntervalMesh(20, L)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    diffusion_params = DiffusionParameters(kappa=0.75, mu=5)
    V = domain.spaces("DG")
    Vu = VectorFunctionSpace(mesh, "CG", 1)

    equation = AdvectionDiffusionEquation(domain, V, "f", Vu=Vu,
                                          diffusion_parameters=diffusion_params)
    spatial_methods = [DGUpwind(equation, "f"),
                       InteriorPenaltyDiffusion(equation, "f", diffusion_params)]

    # I/O
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    io = IO(domain, output)

    # Time stepper
    if timestepper == 'split1':
        dynamics_schemes = {'transport': ImplicitMidpoint(domain),
                            'diffusion': ForwardEuler(domain)}
        term_splitting = ['transport', 'diffusion']
        stepper = SplitTimestepper(equation, term_splitting, dynamics_schemes, io, spatial_methods=spatial_methods)
    else:
        dynamics_schemes = {'transport': SSPRK3(domain),
                            'diffusion': ForwardEuler(domain)}
        term_splitting = ['diffusion', 'transport']
        stepper = SplitTimestepper(equation, term_splitting, dynamics_schemes, io, spatial_methods=spatial_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    x = SpatialCoordinate(mesh)
    xc_init = 0.25*L
    xc_end = 0.75*L
    umax = 0.5*L/tmax

    # Get minimum distance on periodic interval to xc
    x_init = conditional(sqrt((x[0] - xc_init)**2) < 0.5*L,
                         x[0] - xc_init, L + x[0] - xc_init)

    x_end = conditional(sqrt((x[0] - xc_end)**2) < 0.5*L,
                        x[0] - xc_end, L + x[0] - xc_end)

    f_init = 5.0
    f_end = f_init / 2.0
    f_width_init = L / 10.0
    f_width_end = f_width_init * 2.0
    f_init_expr = f_init*exp(-(x_init / f_width_init)**2)
    f_end_expr = f_end*exp(-(x_end / f_width_end)**2)

    stepper.fields('f').interpolate(f_init_expr)
    stepper.fields('u').interpolate(as_vector([Constant(umax)]))
    f_end = stepper.fields('f_end', space=V)
    f_end.interpolate(f_end_expr)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(0, tmax=tmax)

    error = norm(stepper.fields('f') - f_end) / norm(f_end)

    return error


@pytest.mark.parametrize("timestepper", ["split1", "split2"])
def test_split_timestepper_adv_diff(tmpdir, timestepper):

    tol = 0.015
    error = run_split_timestepper_adv_diff(tmpdir, timestepper)
    assert error < tol, 'The error in the advection-diffusion ' + \
        'equation with a split timestepper is greater than ' + \
        'the permitted tolerance'


def run_split_timestepper_sw_evap(dirname, splitting):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Parameters
    dt = 100
    R = 6371220.
    H = 100
    theta_c = pi
    lamda_c = pi/2
    rc = R/4
    beta2 = 10

    # Domain
    mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=2)
    degree = 1
    domain = Domain(mesh, dt, 'BDM', degree)
    x = SpatialCoordinate(mesh)
    lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])

    # saturation field (constant everywhere)
    sat = 100

    # Equation
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R

    tracers = [WaterVapour(space='DG'), CloudWater(space='DG')]

    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                 u_transport_option='vector_advection_form',
                                 thermal=True, active_tracers=tracers)

    # I/O
    output = OutputParameters(dirname=dirname+"/split_timestep_sw_evap",
                              dumpfreq=1)
    io = IO(domain, output,
            diagnostic_fields=[Sum('water_vapour', 'cloud_water')])

    # Physics schemes
    physics_schemes = [(SWSaturationAdjustment(eqns, sat,
                                               parameters=parameters,
                                               thermal_feedback=True,
                                               beta2=beta2),
                        ForwardEuler(domain))]

    # Timestepper
    if splitting == 'split1':
        dynamics_schemes = {'transport': SSPRK3(domain),
                            'coriolis': SSPRK3(domain),
                            'pressure_gradient': SSPRK3(domain)}
        term_splitting = ['transport', 'coriolis', 'pressure_gradient', 'physics']
        stepper = SplitTimestepper(eqns, term_splitting, dynamics_schemes, io, physics_schemes=physics_schemes)
    else:
        dynamics_schemes = {'transport': ImplicitMidpoint(domain),
                            'coriolis': SSPRK3(domain),
                            'pressure_gradient': SSPRK3(domain)}
        term_splitting = ['coriolis', 'pressure_gradient', 'physics', 'transport']
        stepper = SplitTimestepper(eqns, term_splitting, dynamics_schemes, io, physics_schemes=physics_schemes)

    # Initial conditions
    b0 = stepper.fields("b")
    v0 = stepper.fields("water_vapour")
    c0 = stepper.fields("cloud_water")

    # perturbation
    r = R * (
        acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_c)))
    pert = conditional(r < rc, 1.0, 0.0)

    # atmosphere is subsaturated and cloud is present
    v0.interpolate(0.96*Constant(sat))
    c0.interpolate(0.005*sat*pert)
    # lose cloud and add this to vapour
    v_true = Function(v0.function_space()).interpolate(sat*(0.96+0.005*pert))
    c_true = Function(c0.function_space()).interpolate(Constant(0.0))
    # gain buoyancy
    factor = parameters.g*beta2
    sat_adj_expr = (v0 - sat) / dt
    sat_adj_expr = conditional(sat_adj_expr < 0,
                               max_value(sat_adj_expr, -c0 / dt),
                               min_value(sat_adj_expr, v0 / dt))
    # include factor of -1 in true solution to compare term to LHS in Gusto
    b_true = Function(b0.function_space()).interpolate(-dt*sat_adj_expr*factor)

    c_init = Function(c0.function_space()).interpolate(c0)

    # Run
    stepper.run(t=0, tmax=dt)

    return eqns, stepper, v_true, c_true, b_true, c_init


@pytest.mark.parametrize("splitting", ["split1", "split2"])
def test_split_timestepper_evap(tmpdir, splitting):

    dirname = str(tmpdir)
    eqns, stepper, v_true, c_true, b_true, c_init = run_split_timestepper_sw_evap(dirname, splitting)

    vapour = stepper.fields("water_vapour")
    cloud = stepper.fields("cloud_water")
    buoyancy = stepper.fields("b")

    # Check that the water vapour is correct
    assert norm(vapour - v_true) / norm(v_true) < 0.001, \
        f'Final vapour field is incorrect with the split timestepper'

    # Check that cloud has been created/removed
    assert norm(cloud - norm(c_true)) / norm(c_init) < 0.001, \
        f'Final cloud field is incorrect with the split timestepper'

    # Check that the buoyancy perturbation has the correct sign and size
    assert norm(buoyancy - b_true) / norm(b_true) < 0.001, \
        f'Latent heating is incorrect for with the split timestepper'

    # Check that total moisture conserved
    filename = join(dirname, "split_timestep_sw_evap/diagnostics.nc")
    data = Dataset(filename, "r")
    water = data.groups["water_vapour_plus_cloud_water"]
    total = water.variables["total"]
    water_t_0 = total[0]
    water_t_T = total[-1]

    assert abs(water_t_0 - water_t_T) / water_t_0 < 1e-12, \
        f'Total amount of water should be conserved by the split timestepper'
    

def run_boussinesq(tmpdir, compressible):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    dt = 6.0
    tmax = 2*dt
    nlayers = 10  # horizontal layers
    ncols = 10  # number of columns
    Lx = 1000.0
    Lz = 1000.0
    mesh_name = 'boussinesq_mesh'
    m = PeriodicIntervalMesh(ncols, Lx)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers, name=mesh_name)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    parameters = BoussinesqParameters()
    eqn = BoussinesqEquations(domain, parameters, compressible=compressible)

    # I/O
    if compressible:
        output_dirname = tmpdir+"/boussinesq_compressible"
    else:
        output_dirname = tmpdir+"/boussinesq_incompressible"
    output = OutputParameters(dirname=output_dirname,
                              dumpfreq=2, chkptfreq=2, checkpoint=True)
    io = IO(domain, output)

    # Transport Schemes
    b_opts = SUPGOptions()
    if compressible:
        transported_fields = [TrapeziumRule(domain, "u"),
                              SSPRK3(domain, "p"),
                              SSPRK3(domain, "b", options=b_opts)]
        transport_methods = [DGUpwind(eqn, "u"),
                             DGUpwind(eqn, "p"),
                             DGUpwind(eqn, "b", ibp=b_opts.ibp)]
    else:
        transported_fields = [TrapeziumRule(domain, "u"),
                              SSPRK3(domain, "b", options=b_opts)]
        transport_methods = [DGUpwind(eqn, "u"),
                             DGUpwind(eqn, "b", ibp=b_opts.ibp)]

    # Linear solver
    linear_solver = BoussinesqSolver(eqn)

    # Time stepper
    if compressible:
        dynamics_schemes = {'transport': ImplicitMidpoint(domain),
                            'gravity': ImplicitMidpoint(domain),
                            'pressure_gradient': ImplicitMidpoint(domain),
                            'divergence': ImplicitMidpoint(domain)}
        term_splitting = ['divergence', 'transport', 'gravity', 'pressure_gradient']
        stepper = SplitTimestepper(eqn, term_splitting, dynamics_schemes, io)
    else:
        dynamics_schemes = {'transport': SSPRK3(domain),
                            'gravity': SSPRK3(domain),
                            'pressure_gradient': SSPRK3(domain),
                            'incompressible': SSPRK3(domain)}
        term_splitting = ['transport', 'gravity', 'pressure_gradient', 'incompressible']
        stepper = SplitTimestepper(eqn, term_splitting, dynamics_schemes, io)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    p0 = stepper.fields("p")
    b0 = stepper.fields("b")
    u0 = stepper.fields("u")

    # Add horizontal translation to ensure some transport happens
    u0.project(as_vector([0.5, 0.0]))

    # z.grad(bref) = N**2
    x, z = SpatialCoordinate(mesh)
    N = parameters.N
    bref = z*(N**2)

    b_b = Function(b0.function_space()).interpolate(bref)
    boussinesq_hydrostatic_balance(eqn, b_b, p0)
    stepper.set_reference_profiles([('p', p0), ('b', b_b)])

    # Add perturbation
    r = sqrt((x-Lx/2)**2 + (z-Lz/2)**2)
    b_pert = 0.1*exp(-(r/(Lx/5)**2))
    b0.interpolate(b_b + b_pert)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)

    # State for checking checkpoints
    if compressible:
        checkpoint_name = 'compressible_boussinesq_chkpt.h5'
    else:
        checkpoint_name = 'incompressible_boussinesq_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=output_dirname,
                                    checkpoint_pickup_filename=new_path,
                                    checkpoint=True)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, "CG", 1)
    check_eqn = BoussinesqEquations(check_domain, parameters, compressible)
    check_io = IO(check_domain, check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [], [])
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)

    return stepper, check_stepper


@pytest.mark.parametrize("compressible", [True, False])
def test_boussinesq(tmpdir, compressible):

    dirname = str(tmpdir)
    stepper, check_stepper = run_boussinesq(dirname, compressible)

    for variable in ['u', 'b', 'p']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        diff_array = new_variable.dat.data - check_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(check_variable.dat.data)

        if compressible:
            compressible_test = 'compressible'
        else:
            compressible_test = 'incompressible'

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            '{compressible_test} test do not match KGO values'
            
            
def run_moist_compressible(tmpdir):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    dt = 6.0
    tmax = 2*dt
    nlayers = 10  # horizontal layers
    ncols = 10  # number of columns
    Lx = 1000.0
    Lz = 1000.0
    mesh_name = 'moist_compressible_mesh'
    m = PeriodicIntervalMesh(ncols, Lx)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers, name=mesh_name)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    parameters = CompressibleParameters()
    tracers = [WaterVapour(name='vapour_mixing_ratio'), CloudWater(name='cloud_liquid_mixing_ratio')]
    eqn = CompressibleEulerEquations(domain, parameters, active_tracers=tracers)

    # I/O
    output = OutputParameters(dirname=tmpdir+"/moist_compressible",
                              dumpfreq=2, checkpoint=True, chkptfreq=2)
    io = IO(domain, output)

    # Transport schemes
    transport_methods = [DGUpwind(eqn, "u"),
                         DGUpwind(eqn, "rho"),
                         DGUpwind(eqn, "theta")]

    # Linear solver
    linear_solver = CompressibleSolver(eqn)

    # Time stepper
    dynamics_schemes = {'transport': SSPRK3(domain),
                        'gravity': SSPRK3(domain),
                        'coriolis': SSPRK3(domain),
                        'pressure_gradient': SSPRK3(domain)}
    term_splitting = ['transport', 'gravity', 'coriolis', 'pressure_gradient']
    stepper = SplitTimestepper(eqn, term_splitting, dynamics_schemes, io)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    R_d = parameters.R_d
    R_v = parameters.R_v
    g = parameters.g

    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")
    m_v0 = stepper.fields("vapour_mixing_ratio")
    u0 = stepper.fields("u")

    # Add horizontal translation to ensure some transport happens
    u0.project(as_vector([0.5, 0.0]))

    # Approximate hydrostatic balance
    x, z = SpatialCoordinate(mesh)
    T = Constant(300.0)
    m_v0.interpolate(Constant(0.01))
    T_vd = T * (1 + R_v * m_v0 / R_d)
    zH = R_d * T / g
    p = Constant(100000.0) * exp(-z / zH)
    theta0.interpolate(tde.theta(parameters, T_vd, p))
    rho0.interpolate(p / (R_d * T))

    stepper.set_reference_profiles([('rho', rho0), ('theta', theta0),
                                    ('vapour_mixing_ratio', m_v0)])

    # Add perturbation
    r = sqrt((x-Lx/2)**2 + (z-Lz/2)**2)
    theta_pert = 1.0*exp(-(r/(Lx/5))**2)
    theta0.interpolate(theta0 + theta_pert)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)

    # State for checking checkpoints
    checkpoint_name = 'moist_compressible_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=tmpdir+"/moist_compressible",
                                    checkpoint_pickup_filename=new_path,
                                    checkpoint=True)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, "CG", 1)
    check_eqn = CompressibleEulerEquations(check_domain, parameters, active_tracers=tracers)
    check_io = IO(check_domain, output=check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [], [])
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)

    return stepper, check_stepper


def test_moist_compressible(tmpdir):

    dirname = str(tmpdir)
    stepper, check_stepper = run_moist_compressible(dirname)

    for variable in ['u', 'rho', 'theta', 'vapour_mixing_ratio']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        diff_array = new_variable.dat.data - check_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(check_variable.dat.data)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'Moist Compressible test do not match KGO values'

