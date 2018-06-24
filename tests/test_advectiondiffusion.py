from os import path
from gusto import *
from firedrake import as_vector, Constant, PeriodicIntervalMesh, \
    SpatialCoordinate, ExtrudedMesh, FunctionSpace, Function, \
    conditional, sqrt, FiniteElement, TensorProductElement, BrokenElement, interval, cos, sin
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from netCDF4 import Dataset
from math import pi

# This tests the AdvectionDiffusion timestepper, by checking that
# profiles are actually advected by it.

def setup_advection_diffusion(dirname):

    # declare grid shape
    L = 800.
    H = 800.
    ncolumns = int(L / 20.)
    nlayers = int(H / 20.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x, z = SpatialCoordinate(mesh)

    fieldlist = ['u','rho']
    timestepping = TimesteppingParameters(dt=1.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/advection_diffusion",
                              dumpfreq=5,
                              dumplist=['u', 'rho', 'tracer'],
                              perturbation_fields=['rho', 'tracer'])
    parameters = CompressibleParameters()

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist)


    # declare initial fields
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    tracer0 = state.fields("tracer", rho0.function_space())

    Vu = u0.function_space()
    Vr = rho0.function_space()
    Vpsi = FunctionSpace(mesh, "CG", 2)

    # Isentropic background state
    xc = Constant(L / 2)
    zc = Constant(H / 2)
    rc = Constant(L / 5)
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    expr = conditional(r < rc, Constant(1.0) + 0.1 * (cos(pi * r / (2 * rc))) ** 2, Constant(1.0))
    
    rho0.interpolate(expr)
    tracer0.interpolate(expr)
    rho_b = Function(Vr).interpolate(Constant(1.0))
    tracer_b = Function(Vr).interpolate(Constant(1.0))

    # set up velocity field
    u_max = Constant(5.0)
    psi_expr = (-u_max * L / pi) * sin(2 * pi * x / L) * sin(pi * z / L)

    psi0 = Function(Vpsi).interpolate(psi_expr)
    gradperp = lambda u: as_vector([-u.dx(1), u.dx(0)])
    u0.project(gradperp(psi0))

    state.initialise([('u', u0), ('rho', rho0), ('tracer', tracer0)])
    state.set_reference_profiles([('rho', rho0), ('tracer', tracer0)])

    # set up advection schemes
    rhoeqn = EmbeddedDGAdvection(state, Vr, equation_form="continuity")
    tracereqn = EmbeddedDGAdvection(state, Vr, equation_form="advective")

    # build advection dictionary
    advected_fields = []
    advected_fields.append(('u', NoAdvection(state, u0, None)))
    advected_fields.append(('rho', SSPRK3(state, rho0, rhoeqn)))
    advected_fields.append(('tracer', SSPRK3(state, tracer0, tracereqn)))

    # build time stepper
    stepper = AdvectionDiffusion(state, advected_fields)

    return stepper, 40.0


def run_advection_diffusion(dirname):

    stepper, tmax = setup_advection_diffusion(dirname)
    stepper.run(t=0, tmax=tmax)


def test_advection_diffusion_setup(tmpdir):

    dirname = str(tmpdir)
    run_advection_diffusion(dirname)
    filename = path.join(dirname, "advection_diffusion/diagnostics.nc")
    data = Dataset(filename, "r")

    tracer = data.groups["tracer_perturbation"]
    l2_tracer = tracer.variables["l2"]

    rho = data.groups["rho_perturbation"]
    l2_rho = rho.variables["l2"]

    tolerance = 1e-5

    # check that the fields have indeed been advected
    # tracer represents a passive tracer
    assert l2_tracer[-1] >= tolerance
    # rho represents an "active" field
    assert l2_rho[-1] >= tolerance

