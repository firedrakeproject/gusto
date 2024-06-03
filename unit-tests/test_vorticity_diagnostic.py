
from gusto.diagnostics import RelativeVorticity
from gusto.fields import StateFields, PrescribedFields, TimeLevelFields
from gusto import Domain, CompressibleParameters, CompressibleEulerEquations
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, Function, sin, cos,
                       SpatialCoordinate, pi, as_vector, errornorm, norm)


def test_vorticity():
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
    print(error)
    # We dont expect it to be zero as the discrete vorticity is not equal to analytic and dependent on resolution
    assert error < 1e-6, \
        'Relative Vorticity not in error tolerence'
