from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       Constant, DirichletBC, pi, cos, Function, sqrt,
                       conditional, FunctionSpace, VectorFunctionSpace, BrokenElement)
import sys

if '--running-tests' in sys.argv:
    res_dt = {800.: 4.}
    tmax = 4.
else:
    res_dt = {800.: 4., 400.: 2., 200.: 1., 100.: 0.5, 50.: 0.25}
    tmax = 15.*60.

recovered = True if '--recovered' in sys.argv else False
degree = 0 if recovered else 1

L = 51200.

# build volume mesh
H = 6400.  # Height position of the model top

for delta, dt in res_dt.items():

    dirname = "db_dx%s_dt%s" % (delta, dt)
    if recovered:
        dirname += '_recovered'
    nlayers = int(H/delta)  # horizontal layers
    columns = int(L/delta)  # number of columns

    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)

    output = OutputParameters(dirname=dirname,
                              dumpfreq=5,
                              dumplist=['u'],
                              perturbation_fields=['theta', 'rho'],
                              log_level='INFO')

    parameters = CompressibleParameters()
    diagnostics = Diagnostics(*fieldlist)
    diagnostic_fields = [CourantNumber()]

    state = State(mesh, vertical_degree=degree, horizontal_degree=degree,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  fieldlist=fieldlist,
                  diagnostic_fields=diagnostic_fields)

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()
    Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))

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

    physics_boundary = Boundary_Method.physics if recovered else None
    rho_b_Vt = Function(Vt)
    rho_b_recoverer = Recoverer(rho_b, rho_b_Vt, VDG=Vt_brok, boundary_method=physics_boundary).project()
    pie = Function(Vt).assign(thermodynamics.pi(parameters, rho_b_Vt, theta_b))
    T_pert = conditional(r > 1., 0., -7.5*(1.+cos(pi*r)))
    theta0.interpolate(theta_b + T_pert/pie)
    rho0.assign(rho_b)

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0)])
    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    # Set up advection schemes
    if recovered:
        VDG1 = FunctionSpace(mesh, "DG", 1)
        VCG1 = FunctionSpace(mesh, "CG", 1)
        Vu_DG1 = VectorFunctionSpace(mesh, "DG", 1)
        Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)
        Vu_brok = FunctionSpace(mesh, BrokenElement(Vu.ufl_element()))

        u_opts = RecoveredOptions(embedding_space=Vu_DG1,
                                  recovered_space=Vu_CG1,
                                  broken_space=Vu_brok,
                                  boundary_method=Boundary_Method.dynamics)
        rho_opts = RecoveredOptions(embedding_space=VDG1,
                                    recovered_space=VCG1,
                                    broken_space=Vr,
                                    boundary_method=Boundary_Method.dynamics)
        theta_opts = RecoveredOptions(embedding_space=VDG1,
                                      recovered_space=VCG1,
                                      broken_space=Vt_brok,
                                      boundary_method=Boundary_Method.dynamics)

        ueqn = EmbeddedDGAdvection(state, Vu, equation_form="advective", options=u_opts)
        rhoeqn = EmbeddedDGAdvection(state, Vr, equation_form="continuity", options=rho_opts)
        thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective", options=theta_opts)
    else:
        ueqn = EulerPoincare(state, Vu)
        rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
        supg = True
        if supg:
            thetaeqn = SUPGAdvection(state, Vt, equation_form="advective")
        else:
            thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective", options=EmbeddedDGOptions())

    advected_fields = [('rho', SSPRK3(state, rho0, rhoeqn)),
                       ('theta', SSPRK3(state, theta0, thetaeqn))]
    if recovered:
        advected_fields.append(('u', SSPRK3(state, u0, ueqn)))
    else:
        advected_fields.append(('u', ThetaMethod(state, u0, ueqn)))

    # Set up linear solver
    linear_solver = CompressibleSolver(state)

    # Set up forcing
    compressible_forcing = CompressibleForcing(state)

    if recovered:
        v_bcs = [DirichletBC(VDG1, 0.0, "bottom"),
                 DirichletBC(VDG1, 0.0, "top")]

        project_back_bcs = [DirichletBC(Vu, 0.0, "bottom"),
                            DirichletBC(Vu, 0.0, "top")]
    else:
        bcs = [DirichletBC(Vu, 0.0, "bottom"),
               DirichletBC(Vu, 0.0, "top")]

    if recovered:
        diffused_fields = [("u", RecoveredDiffusion(state, [InteriorPenalty(state, VDG1, kappa=75., mu=Constant(10./delta), bcs=None),
                                                            InteriorPenalty(state, VDG1, kappa=75., mu=Constant(10./delta), bcs=v_bcs)],
                                                    Vu, u_opts, projection_bcs=project_back_bcs)),
                           ("theta", RecoveredDiffusion(state, InteriorPenalty(state, VDG1, kappa=75., mu=Constant(10./delta)),
                                                        Vt, theta_opts))]
    else:
        diffused_fields = [("u", InteriorPenalty(state, Vu, kappa=75.,
                                                 mu=Constant(10./delta), bcs=bcs)),
                           ("theta", InteriorPenalty(state, Vt, kappa=75.,
                                                     mu=Constant(10./delta)))]

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            compressible_forcing, diffused_fields)

    stepper.run(t=0, tmax=tmax)
