from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, Constant, DirichletBC,
                       cos, pi, Function, sqrt, conditional)
import sys

if '--running-tests' in sys.argv:
    res_dt = {800.: 4.}
    tmax = 4.
    dumpfreq = 1
else:
    res_dt = {800.: 4., 400.: 2., 200.: 1., 100.: 0.5, 50.: 0.25}
    tmax = 15.*60.
    dumpfreq = int(tmax / (4*dt))


L = 51200.

# build volume mesh
H = 6400.  # Height position of the model top

for delta, dt in res_dt.items():

    dirname = "tracer_dx%s_dt%s" % (delta, dt)
    nlayers = int(H/delta)  # horizontal layers
    columns = int(L/delta)  # number of columns

    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    output = OutputParameters(dirname=dirname,
                              dumpfreq=dumpfreq,
                              dumplist=['u'],
                              perturbation_fields=['theta', 'rho'],
                              log_level='INFO')

    parameters = CompressibleParameters()
    diagnostic_fields = [CourantNumber()]

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    diffusion_options = [
        ("u", DiffusionParameters(kappa=75., mu=10./delta)),
        ("theta", DiffusionParameters(kappa=75., mu=10./delta))]
    eqns = CompressibleEulerEquations(state, "CG", 1,
                                      diffusion_options=diffusion_options)
    watereqn = AdvectionEquation(state, state.spaces("theta"), "water")

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")
    water0 = state.fields("water")

    # spaces
    Vu = state.spaces("HDiv")
    Vt = state.spaces("theta")
    Vr = state.spaces("DG")

    # Isentropic background state
    Tsurf = Constant(300.)

    theta_b = Function(Vt).interpolate(Tsurf)
    rho_b = Function(Vr)

    # Calculate hydrostatic Pi
    compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)

    x = SpatialCoordinate(mesh)
    a = 5.0e3
    deltaTheta = 1.0e-2
    xc = 0.5*L
    xr = 4000.
    zc = 3000.
    zr = 2000.
    r = sqrt(((x[0]-xc)/xr)**2 + ((x[1]-zc)/zr)**2)
    theta_pert = conditional(r > 1., 0., -7.5*(1.+cos(pi*r)))
    theta0.interpolate(theta_b + theta_pert)
    water0.interpolate(theta_pert)
    rho0.assign(rho_b)

    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    # Set up transport schemes
    supg = True
    if supg:
        theta_opts = SUPGOptions()
        water_opts = SUPGOptions()
    else:
        theta_opts = EmbeddedDGOptions()
        water_opts = EmbeddedDGOptions()
    transported_fields = [ImplicitMidpoint(state, "u"),
                          SSPRK3(state, "rho"),
                          SSPRK3(state, "theta", options=theta_opts)]

    # Set up linear solver
    linear_solver = CompressibleSolver(state, eqns)

    bcs = [DirichletBC(Vu, 0.0, "bottom"),
           DirichletBC(Vu, 0.0, "top")]
    diffusion_schemes = [BackwardEuler(state, "u"),
                         BackwardEuler(state, "theta")]

    # build time stepper
    stepper = CrankNicolson(state, eqns, transported_fields,
                            auxiliary_equations_and_schemes=[(watereqn, SSPRK3(state, options=water_opts))],
                            linear_solver=linear_solver,
                            diffusion_schemes=diffusion_schemes)

    stepper.run(t=0, tmax=tmax)
