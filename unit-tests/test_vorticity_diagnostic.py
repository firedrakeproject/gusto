
from gusto.diagnostics import RelativeVorticity, ZonalComponent, MeridionalComponent, RadialComponent
from gusto.fields import StateFields, PrescribedFields, TimeLevelFields
from gusto import (Domain, CompressibleParameters, CompressibleEulerEquations, 
                   GeneralCubedSphereMesh, lonlatr_from_xyz, xyz_vector_from_lonlatr)
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, Function, sin, cos, File,
                       SpatialCoordinate, pi, as_vector, errornorm, norm)
import pytest


@pytest.mark.parametrize("topology", ["slice", "sphere"])
def test_relative_vorticity(topology):
    if topology=='slice':
        vertical_slice_test()
    
    if topology=='sphere':
        sphere_test()
    

def sphere_test():
    R = 1 # radius of ball
    H = 5 # height of model top
    nlayers=5
    c=8
    # Building model and state object
    m = GeneralCubedSphereMesh(R, num_cells_per_edge_of_panel=c, degree=2)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers, extrusion_type='radial')
    domain=Domain(mesh, 0.1, 'RTCF',  degree=1)
    params = CompressibleParameters(g=0, cp=0)
    eqn = CompressibleEulerEquations(domain, params)
    prog_field = TimeLevelFields(eqn)
    HDiv = domain.spaces('HDiv')
    HCurl = domain.spaces('HCurl')
    pres_field = PrescribedFields()
    pres_field('u', HDiv)
    state = StateFields(prog_field, pres_field)

    # Getting spherical co-ordinates
    xyz = SpatialCoordinate(mesh)
    _, lat, r = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
    e_zonal = xyz_vector_from_lonlatr(1, 0, 0, xyz)
    e_rad = xyz_vector_from_lonlatr(0, 0, 1, xyz)


    # Initlising vorticity field
    zonal_u = sin(lat) / r
    merid_u = 0.0
    radial_u = 0.0
    #u_expr = xyz_vector_from_lonlatr(zonal_u, merid_u, radial_u, xyz)
    print('Projecting u')
    state.u.project(e_zonal*zonal_u)

    # Analytic relative vorticity
    radial_vort = 2*cos(lat) / r**2
    analytical_vort_expr = xyz_vector_from_lonlatr(0, 0, radial_vort, xyz)
    print('Projecting analytical vorticity')
    vorticity_analytic = Function(HCurl, name='exact_vorticity').project(e_rad*radial_vort)

    # Setup and compute diagnostic vorticity and zonal components
    Vorticity = RelativeVorticity()
    Vorticity.setup(domain, state)
    print('diagnosing vorticity')
    Vorticity.compute()

    Zonal_u = ZonalComponent('u')
    Zonal_u.setup(domain, state)
    Zonal_u.compute()

    Meridional_u = MeridionalComponent('u')
    Meridional_u.setup(domain, state)
    Meridional_u.compute()

    Radial_u = RadialComponent('u')
    Radial_u.setup(domain, state)
    Radial_u.compute()


  #  zonal_diff = Function(HDiv, name='zonal_diff').project(state.u - state.u_zonal) 

    diff = Function(HCurl, name='difference').project(vorticity_analytic - state.RelativeVorticity)


    radial_diagnostic_vort = RadialComponent('RelativeVorticity')
    radial_diagnostic_vort.setup(domain, state)
    radial_diagnostic_vort.compute()

    # Compare analytic vorticity expression to diagnostic
    print('calculating error')
    error = errornorm(vorticity_analytic, state.RelativeVorticity) / norm(vorticity_analytic)
    print(error)

    outfile = File('spherical_vorticity.pvd')
    outfile.write(state.u, vorticity_analytic, state.RelativeVorticity, diff, state.u_zonal, state.u_meridional, state.u_radial, state.RelativeVorticity_radial)
    # We dont expect it to be zero as the discrete vorticity is not equal to analytic and dependent on resolution
    assert error < 1e-6, \
        'Relative Vorticity not in error tolerence'

def vertical_slice_test():
    L = 10
    H = 10
    ncol = 100
    nlayers = 100

    m = PeriodicIntervalMesh(ncol, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    _, z = SpatialCoordinate(mesh)

    domain = Domain(mesh, 0.1, 'CG', 1)
    params = CompressibleParameters()
    eqn = CompressibleEulerEquations(domain, params)
    prog_field = TimeLevelFields(eqn)

    H1 = domain.spaces('H1')
    HDiv = domain.spaces('HDiv')

    u_expr = 3 * sin(2*pi*z/H)
    vort_exact_expr = -6*pi/H * cos(2*pi*z/H)
    vorticity_analytic = Function(H1, name='analytic_vort').interpolate(vort_exact_expr)

    # Setting up test field for the diagnostic to use
    prescribed_fields = PrescribedFields()
    prescribed_fields('u', HDiv)
    state = StateFields(prog_field, prescribed_fields)
    state.u.project(as_vector([u_expr, 0]))

    Vorticity = RelativeVorticity()
    Vorticity.setup(domain, state)
    Vorticity.compute()
    # Compare analytic vorticity expression to diagnostic
    error = errornorm(vorticity_analytic, state.RelativeVorticity) / norm(vorticity_analytic)
    # We dont expect it to be zero as the discrete vorticity is not equal to analytic and dependent on resolution
    assert error < 1e-6, \
        'Relative Vorticity not in error tolerence'

sphere_test()