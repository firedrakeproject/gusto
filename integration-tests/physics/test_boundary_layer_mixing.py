"""
This tests the physics routine to mix fields in the boundary layer.
"""

from gusto import *
from gusto.labels import physics_label
from firedrake import (VectorFunctionSpace, PeriodicIntervalMesh, as_vector,
                       exp, SpatialCoordinate, ExtrudedMesh, Function)
from firedrake.fml import identity
import pytest


def run_boundary_layer_mixing(dirname, field_name, recovered, semi_implicit):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    element_degree = 1 if field_name == 'u' and not recovered else 0
    dt = 100.0

    # declare grid shape, with length L and height H
    L = 500.
    H = 500.
    nlayers = 5
    ncolumns = 3

    # make mesh and domain
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    domain = Domain(mesh, dt, "CG", element_degree)

    _, z = SpatialCoordinate(mesh)

    # Set up equation
    parameters = CompressibleParameters()
    eqn = CompressibleEulerEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=dirname+"/boundary_layer_mixing",
                              dumpfreq=1,
                              dumplist=[field_name])
    io = IO(domain, output)

    # Physics scheme
    surf_params = BoundaryLayerParameters()
    physics_parametrisation = BoundaryLayerMixing(eqn, field_name, surf_params)

    if recovered:
        # Only implemented for u
        Vec_CG1 = VectorFunctionSpace(mesh, 'CG', 1)
        Vec_CG1 = VectorFunctionSpace(mesh, 'DG', 1)
        recovery_opts = RecoveryOptions(embedding_space=Vec_CG1,
                                        recovered_space=Vec_CG1,
                                        boundary_method=BoundaryMethod.taylor)
        implicit_discretisation = BackwardEuler(domain, field_name=field_name, options=recovery_opts)
    else:
        implicit_discretisation = BackwardEuler(domain)

    if semi_implicit:
        if recovered:
            explicit_discretisation = ForwardEuler(domain, field_name=field_name, options=recovery_opts)
        else:
            explicit_discretisation = ForwardEuler(domain)

        # Use half of the time discretisation for each
        explicit_discretisation.dt.assign(domain.dt/2)
        implicit_discretisation.dt.assign(domain.dt/2)
        physics_schemes = [(physics_parametrisation, explicit_discretisation),
                           (physics_parametrisation, implicit_discretisation)]
    else:
        physics_schemes = [(physics_parametrisation, implicit_discretisation)]

    # Only want time derivatives and physics terms in equation, so drop the rest
    eqn.residual = eqn.residual.label_map(lambda t: any(t.has_label(time_derivative, physics_label)),
                                          map_if_true=identity, map_if_false=drop)

    # Time stepper
    scheme = ForwardEuler(domain)
    stepper = SplitPhysicsTimestepper(eqn, scheme, io,
                                      physics_schemes=physics_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    Vt = domain.spaces("theta")
    Vu = domain.spaces("HDiv")

    # Declare prognostic fields
    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # Set prognostic variables
    theta0.interpolate(300.*exp(-z/(2*H)))
    rho0.interpolate(1.1*exp(-z/(5*H)))

    u0.project(as_vector([5.0*exp(-z/(0.5*H)), 0.0]))

    if field_name == 'theta':
        initial_field = Function(Vt)
        initial_field.assign(theta0)
    elif field_name == 'u':
        initial_field = Function(Vu)
        initial_field.assign(u0)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt)

    return domain, stepper, initial_field


@pytest.mark.parametrize("field_name, recovered, semi_implicit",
                         [('theta', False, False),
                          ('theta', False, True),
                          ('u', False, False),
                          pytest.param('u', True, True,
                                       marks=pytest.mark.xfail(reason='recovered physics not implemented'))
                          ])
def test_boundary_layer_mixing(tmpdir, field_name, recovered, semi_implicit):

    dirname = str(tmpdir)
    domain, stepper, initial_field = \
        run_boundary_layer_mixing(dirname, field_name, recovered, semi_implicit)

    if field_name == 'u':
        # Need to project horizontal wind into W3
        wind_2d = stepper.fields(field_name)
        field = Function(domain.spaces('L2')).project(wind_2d[0])
        initial_1d = Function(domain.spaces('L2')).project(initial_field[0])
        # Relabel initial field
        initial_field = initial_1d
    else:
        field = stepper.fields(field_name)

    field_data, _ = domain.coords.get_column_data(field, domain)
    initial_data, _ = domain.coords.get_column_data(initial_field, domain)

    # Check first column
    assert field_data[0, 0] < 0.999*initial_data[0, 0]
