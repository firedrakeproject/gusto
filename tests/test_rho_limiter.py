from gusto import *
from firedrake import as_vector, Constant, sin, PeriodicIntervalMesh, \
    SpatialCoordinate, ExtrudedMesh, Expression, VertexBasedLimiter
import json
from math import pi

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then advected by a prescribed advection scheme


def setup_rho_limiter(dirname):

    # declare grid shape, with length L and height H
    L = 1000.
    H = 400.
    nlayers = int(H / 20.)
    ncolumns = int(L / 20.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x,z = SpatialCoordinate(mesh)

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=1.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/rho_limiter",
                              dumpfreq=5,
                              dumplist=['u'],
                              perturbation_fields=['rho'])
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

    # spaces
    Vpsi = FunctionSpace(mesh, "CG", 2)
    Vr = rho0.function_space()

    # make a gradperp
    gradperp = lambda u: as_vector([-u.dx(1), u.dx(0)])

    # Isentropic background state
    rho_surf = 1.
    rhob = Constant(rho_surf)
    rho_b = Function(Vr).interpolate(rhob)

    # set up bubble
    xc = 200.
    zc = 200.
    rc = 100.
    rho_pert = Function(Vr).interpolate(conditional(sqrt((x - xc) ** 2.0) < rc,
                                                    conditional(sqrt((z - zc) ** 2.0) < rc,
                                                                Constant(0.2),
                                                                Constant(0.0)), Constant(0.0)))

    # set up velocity field
    u_max = Constant(1.0)

    psi_expr = - u_max * z

    psi0 = Function(Vpsi).interpolate(psi_expr)
    u0.project(gradperp(psi0))
    rho0.interpolate(rho_b + rho_pert)

    state.initialise({'u': u0, 'rho': rho0})
    state.set_reference_profiles({'rho': rho_b})

    # set up advection schemes
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")

    # build advection dictionary
    advection_dict = {}
    advection_dict["u"] = NoAdvection(state, u0, None)
    advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn, limiter=VertexBasedLimiter(rho0.function_space()))

    # forcing
    forcing = NoForcing(state)

    # linear solver
    linear_solver = NoSolver(state)

    # build time stepper
    stepper = Timestepper(state, advection_dict)

    return stepper, 300.0


def run_rho_limiter(dirname):

    stepper, tmax = setup_rho_limiter(dirname)
    stepper.run(t=0, tmax=tmax)


def test_rho_limiter_setup(tmpdir):

    dirname = str(tmpdir)
    run_rho_limiter(dirname)
