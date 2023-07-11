"""
Tests gusto's netCDF outputting capability. An advection equation is solved on
different domains with the netCDF outputting turned on to test that the model
does not fail on these domains. The outputted metadata is checked.
"""

from firedrake import (IntervalMesh, RectangleMesh, CubedSphereMesh,
                       VectorFunctionSpace, ExtrudedMesh, SpatialCoordinate,
                       as_vector, exp)
from gusto import (Domain, IO, PrescribedTransport, AdvectionEquation,
                   ForwardEuler, OutputParameters, VelocityX, VelocityY,
                   VelocityZ, MeridionalComponent, ZonalComponent,
                   RadialComponent, DGUpwind)
from netCDF4 import Dataset
import pytest


@pytest.fixture
def domain_and_mesh_details(geometry):

    dt = 0.001
    Lx = 3.1
    Ly = 5.0
    Hz = 0.4
    radius = 5.0

    if geometry == 'interval':
        mesh, family = IntervalMesh(3, Lx), 'CG'
        mesh_details = {'domain_extent_x': Lx}
    elif geometry == 'plane':
        mesh, family = RectangleMesh(3, 3, Lx, Ly, quadrilateral=True), 'RTCF'
        mesh_details = {'domain_extent_x': Lx, 'domain_extent_y': Ly}
    elif geometry == 'spherical_shell':
        mesh, family = CubedSphereMesh(radius, 2), 'RTCF'
        mesh_details = {}
    elif geometry == 'vertical_slice':
        base_mesh, family = IntervalMesh(3, Lx), 'CG'
        mesh = ExtrudedMesh(base_mesh, 3, Hz/3.0)
        mesh_details = {'domain_extent_x': Lx, 'domain_extent_z': Hz}
    elif geometry == 'extruded_plane':
        base_mesh, family = RectangleMesh(3, 3, Lx, Ly, quadrilateral=True), 'RTCF'
        mesh = ExtrudedMesh(base_mesh, 3, Hz/3.0)
        mesh_details = {'domain_extent_x': Lx, 'domain_extent_y': Ly, 'domain_extent_z': Hz}
    elif geometry == 'extruded_spherical_shell':
        base_mesh, family = CubedSphereMesh(radius, 2), 'RTCF'
        mesh = ExtrudedMesh(base_mesh, 3, Hz/3.0, extrusion_type='radial')
        mesh_details = {'domain_extent_z': Hz}

    domain = Domain(mesh, dt, family, degree=1)
    mesh_details['extruded'] = mesh.extruded
    mesh_details['domain_type'] = geometry

    return (domain, mesh_details)


# TODO: make parallel configurations of this test
@pytest.mark.parametrize("geometry", ["interval", "vertical_slice",
                                      "plane", "extruded_plane",
                                      "spherical_shell", "extruded_spherical_shell"])
def test_nc_outputting(tmpdir, geometry, domain_and_mesh_details):

    # ------------------------------------------------------------------------ #
    # Make model objects
    # ------------------------------------------------------------------------ #

    dirname = str(tmpdir)
    domain, mesh_details = domain_and_mesh_details
    V = domain.spaces('DG')
    if geometry == "interval":
        VecCG1 = VectorFunctionSpace(domain.mesh, "CG", 1)
        eqn = AdvectionEquation(domain, V, 'f', Vu=VecCG1)
    else:
        eqn = AdvectionEquation(domain, V, 'f')
    transport_scheme = ForwardEuler(domain)
    transport_method = DGUpwind(eqn, 'f')
    output = OutputParameters(dirname=dirname, dumpfreq=1, dump_nc=True,
                              dumplist=['f'], log_level='INFO', checkpoint=False)

    # Make velocity components for this geometry
    if geometry == "interval":
        diagnostic_fields = [VelocityX()]
    elif geometry == "vertical_slice":
        diagnostic_fields = [VelocityX(), VelocityZ()]
    elif geometry == "plane":
        diagnostic_fields = [VelocityX(), VelocityY()]
    elif geometry == "extruded_plane":
        diagnostic_fields = [VelocityX(), VelocityY(), VelocityZ()]
    elif geometry == "spherical_shell":
        diagnostic_fields = [ZonalComponent('u'), MeridionalComponent('u')]
    elif geometry == "extruded_spherical_shell":
        diagnostic_fields = [ZonalComponent('u'), MeridionalComponent('u'), RadialComponent('u')]

    io = IO(domain, output, diagnostic_fields=diagnostic_fields)
    stepper = PrescribedTransport(eqn, transport_scheme, transport_method, io)

    # ------------------------------------------------------------------------ #
    # Initialise fields
    # ------------------------------------------------------------------------ #

    xyz = SpatialCoordinate(domain.mesh)

    f = stepper.fields('f')
    u = stepper.fields('u')
    if geometry == "interval":
        fexpr = exp(-xyz[0])
        uexpr = as_vector([0.005])
    elif geometry in ["vertical_slice", "plane"]:
        fexpr = exp(-xyz[0])*exp(-xyz[1])
        uexpr = as_vector([0.005, 0.0])
    elif geometry == "extruded_plane":
        fexpr = exp(-xyz[0])*exp(-xyz[1])
        uexpr = as_vector([0.005, 0.0002, 0.0])
    elif geometry in ["spherical_shell", "extruded_spherical_shell"]:
        fexpr = xyz[0]
        uexpr = as_vector([-xyz[1]/100.0, xyz[0]/100.0, 0.0])

    f.interpolate(fexpr)
    u.project(uexpr)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(0, tmax=3*float(domain.dt))

    # ------------------------------------------------------------------------ #
    # Checking
    # ------------------------------------------------------------------------ #

    # Check that metadata is correct
    output_data = Dataset(f'{dirname}/field_output.nc', 'r')
    for metadata_key, metadata_value in mesh_details.items():
        # Convert None or booleans to strings
        if type(metadata_value) in [type(None), type(True)]:
            output_value = str(metadata_value)
        else:
            output_value = metadata_value

        error_message = f'Metadata {metadata_key} for geometry {geometry} is incorrect'
        if type(output_value) == float:
            assert output_data[metadata_key][0] - output_value < 1e-14, error_message
        else:
            assert output_data[metadata_key][0] == output_value, error_message
