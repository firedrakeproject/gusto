"""
Linear gravity wave test case of Skamarock and Klemp (1994), solved using
REXI applied to the compressible Boussinesq equations.

"""

from os.path import join, abspath, dirname
from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       sin, SpatialCoordinate, Function, pi)
from firedrake.output import VTKFile


def run_rexi_linear_boussinesq(tmpdir):

    # ---------------------------------------------------------------------- #
    # Set up model objects
    # ---------------------------------------------------------------------- #
    tmax = 1000
    L = 1.0e3  # Domain length
    H = 1.0e3  # Height position of the model top

    columns = 10  # number of columns
    nlayers = 10  # horizontal layers

    # Domain
    mesh_name = 'linear_boussinesq_mesh'
    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers,
                        name=mesh_name)
    domain = Domain(mesh, tmax, 'CG', 1)

    # Equation
    parameters = BoussinesqParameters(cs=300)
    eqns = LinearBoussinesqEquations(domain, parameters)

    # ---------------------------------------------------------------------- #
    # Initial conditions
    # ---------------------------------------------------------------------- #

    U_in = Function(eqns.function_space, name="U_in")
    U_expl = Function(eqns.function_space, name="Uexpl")
    u, p, b = U_in.subfunctions

    # spaces
    Vb = b.function_space()
    Vp = p.function_space()

    x, z = SpatialCoordinate(mesh)

    # first setup the background buoyancy profile
    # z.grad(bref) = N**2
    N = parameters.N
    bref = z*(N**2)
    # interpolate the expression to the function
    b_b = Function(Vb).interpolate(bref)

    # setup constants
    a = 5.0e3
    deltab = 1.0e-2
    b_pert = deltab*sin(pi*z/H)/(1 + (x - L/2)**2/a**2)
    # interpolate the expression to the function
    b.interpolate(b_b + b_pert)

    p_b = Function(Vp)
    boussinesq_hydrostatic_balance(eqns, b_b, p_b)
    p.assign(p_b)

    # set the background buoyancy and pressure
    _, p_ref, b_ref = eqns.X_ref.subfunctions
    p_ref.assign(p_b)
    b_ref.assign(b_b)

    # ----------------------------------------------------------------------- #
    # Compute exponential solution
    # ----------------------------------------------------------------------- #
    # REXI output
    rexi_output = VTKFile(str(tmpdir)+"/sk_wave/rexi.pvd")
    u1, p1, b1 = U_expl.subfunctions
    u1.assign(u1)
    p1.assign(p1-p_b)
    b1.assign(b1-b_b)
    rexi_output.write(u1, p1, b1)
    rexi = Rexi(eqns, RexiParameters())
    rexi.solve(U_expl, U_in, tmax)
    u1, p1, b1 = U_expl.subfunctions
    p1 -= p_b
    b1 -= b_b
    rexi_output.write(u1, p1, b1)

    # Checkpointing
    checkpoint_name = 'linear_sk_rexi_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=tmpdir+"/linear_sk",
                                    checkpoint_pickup_filename=new_path,
                                    checkpoint=True)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, tmax, 'CG', 1)
    check_eqn = ShallowWaterEquations(check_domain, parameters, fexpr=fexpr)
    check_io = IO(check_domain, output=check_output)
    check_stepper = Timestepper(check_eqn, RK4(check_domain), check_io)
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)
    usoln = check_stepper.fields("u")
    psoln = check_stepper.fields("p")
    bsoln = check_stepper.fields("b")

    return usoln, psoln, bsoln, u1, p1, b1


def test_rexi_linear_boussinesq(tmpdir):

    dirname = str(tmpdir)
    u, p, b, uexpl, pexpl, bexpl = run_rexi_linear_boussinesq(dirname)

    udiff_arr = uexpl.dat.data - usoln.dat.data
    pdiff_arr = pexpl.dat.data - psoln.dat.data
    bdiff_arr = bexpl.dat.data - bsoln.dat.data

    uerror = np.linalg.norm(udiff_arr) / np.linalg.norm(usoln.dat.data)
    perror = np.linalg.norm(pdiff_arr) / np.linalg.norm(psoln.dat.data)
    berror = np.linalg.norm(bdiff_arr) / np.linalg.norm(bsoln.dat.data)

    # Slack values chosen to be robust to different platforms
    assert uerror < 1e-10, \
        'u values in REXI compressible boussinesq test do not match KGO values'
    assert perror < 1e-10, \
        'p values in REXI compressible boussinesq test do not match KGO values'
    assert berror < 1e-10, \
        'b values in REXI compressible boussinesq test do not match KGO values'

