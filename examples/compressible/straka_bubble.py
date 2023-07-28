"""
The falling cold density current test of Straka et al (1993).

This example runs at a series of resolutions with different time steps.
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       Constant, pi, cos, Function, sqrt,
                       conditional)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

if '--running-tests' in sys.argv:
    res_dt = {800.: 4.}
    tmax = 4.
    ndumps = 1
else:
    res_dt = {800.: 4., 400.: 2., 200.: 1., 100.: 0.5, 50.: 0.25}
    tmax = 15.*60.
    ndumps = 4


L = 51200.

# build volume mesh
H = 6400.  # Height position of the model top

for delta, dt in res_dt.items():

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    nlayers = int(H/delta)  # horizontal layers
    columns = int(L/delta)  # number of columns
    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    parameters = CompressibleParameters()
    u_diffusion_opts = DiffusionParameters(kappa=75., mu=10./delta)
    theta_diffusion_opts = DiffusionParameters(kappa=75., mu=10./delta)
    diffusion_options = [("u", u_diffusion_opts), ("theta", theta_diffusion_opts)]
    eqns = CompressibleEulerEquations(domain, parameters,
                                      diffusion_options=diffusion_options)

    # I/O
    dirname = "straka_dx%s_dt%s" % (delta, dt)
    dumpfreq = int(tmax / (ndumps*dt))
    output = OutputParameters(dirname=dirname,
                              dumpfreq=dumpfreq,
                              dumplist=['u'],
                              log_level='INFO')
    diagnostic_fields = [CourantNumber(), Perturbation('theta'), Perturbation('rho')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    theta_opts = SUPGOptions()
    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "rho"),
                          SSPRK3(domain, "theta", options=theta_opts)]
    transport_methods = [DGUpwind(eqns, "u"),
                         DGUpwind(eqns, "rho"),
                         DGUpwind(eqns, "theta", ibp=theta_opts.ibp)]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Diffusion schemes
    diffusion_schemes = [BackwardEuler(domain, "u"),
                         BackwardEuler(domain, "theta")]
    diffusion_methods = [InteriorPenaltyDiffusion(eqns, "u", u_diffusion_opts),
                         InteriorPenaltyDiffusion(eqns, "theta", theta_diffusion_opts)]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                      spatial_methods=transport_methods+diffusion_methods,
                                      linear_solver=linear_solver,
                                      diffusion_schemes=diffusion_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # spaces
    Vu = domain.spaces("HDiv")
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    # Isentropic background state
    Tsurf = Constant(300.)

    theta_b = Function(Vt).interpolate(Tsurf)
    rho_b = Function(Vr)
    exner = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta_b, rho_b, exner0=exner,
                                     solve_for_rho=True)

    x = SpatialCoordinate(mesh)
    a = 5.0e3
    xc = 0.5*L
    xr = 4000.
    zc = 3000.
    zr = 2000.
    r = sqrt(((x[0]-xc)/xr)**2 + ((x[1]-zc)/zr)**2)
    T_pert = conditional(r > 1., 0., -7.5*(1.+cos(pi*r)))
    theta0.interpolate(theta_b + T_pert*exner)
    rho0.assign(rho_b)

    stepper.set_reference_profiles([('rho', rho_b),
                                    ('theta', theta_b)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)
