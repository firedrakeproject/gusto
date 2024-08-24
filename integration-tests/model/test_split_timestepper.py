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


#@pytest.mark.parametrize("timestepper", ["split1", "split2"])
def test_split_timestepper_adv_diff(tmpdir, timestepper):

    tol = 0.015
    error = run_split_timestepper_adv_diff(tmpdir, timestepper)
    assert error < tol, 'The error in the advection-diffusion ' + \
        'equation with a split timestepper is greater than ' + \
        'the permitted tolerance'


def run_split_timestepper_adv_diff_physics(tmpdir, timestepper):

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
        
    x = SpatialCoordinate(mesh)
        
    # Add a source term to inject mass into the domain.  
    # Without the diffusion, this would simply add 0.1
    # unit of mass equally across the domain             
    source_expression = -Constant(0.1)
    
    physics_schemes = [(SourceSink(equation, "f", source_expression), SSPRK3(domain))]

    # I/O
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    io = IO(domain, output)

    # Time stepper
    if timestepper == 'split1':
        dynamics_schemes = {'transport': ImplicitMidpoint(domain),
                            'diffusion': ForwardEuler(domain)}
        term_splitting = ['transport', 'diffusion', 'physics']
        stepper = SplitTimestepper(equation, term_splitting, dynamics_schemes, io, spatial_methods=spatial_methods, physics_schemes=physics_schemes)
    elif timestepper == 'split2':
        dynamics_schemes = {'transport': SSPRK3(domain),
                            'diffusion': ForwardEuler(domain)}
        term_splitting = ['diffusion', 'physics', 'transport']
        stepper = SplitTimestepper(equation, term_splitting, dynamics_schemes, io, spatial_methods=spatial_methods, physics_schemes=physics_schemes)
    else:
        dynamics_schemes = {'transport': SSPRK3(domain),
                            'diffusion': SSPRK3(domain)}
        term_splitting = ['physics', 'transport', 'diffusion', 'transport']
        weights = [1,0.5,1,0.5]
        #weights = [1, 1./3., 1, 2./3.]
        stepper = SplitTimestepper(equation, term_splitting, dynamics_schemes, io, weights=weights, spatial_methods=spatial_methods, physics_schemes=physics_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

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
    
    # The end Gaussian should be advected by half the domain
    # length, be more spread out due to the dissipation,
    # and includes more mass due to the source term.
    f_end_expr = 0.1 + f_end*exp(-(x_end / f_width_end)**2)

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
    
    
@pytest.mark.parametrize("timestepper", ["split1", "split2", "split3"])
def test_split_timestepper_adv_diff_physics(tmpdir, timestepper):

    tol = 0.015
    error = run_split_timestepper_adv_diff_physics(tmpdir, timestepper)
    print(error)
    assert error < tol, 'The split timestepper in the advection-diffusion' + \
        'equation with source physics has an error greater than ' + \
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


#@pytest.mark.parametrize("splitting", ["split1", "split2"])
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
    
