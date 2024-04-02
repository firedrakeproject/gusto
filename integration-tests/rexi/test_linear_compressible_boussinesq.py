"""
Linear gravity wave test case of Skamarock and Klemp (1994), solved using
REXI applied to the compressible Boussinesq equations.

"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       sin, SpatialCoordinate, Function, pi)
from firedrake.output import VTKFile


def run_rexi_linear_boussinesq(tmpdir):

    # ---------------------------------------------------------------------- #
    # Set up model objects
    # ---------------------------------------------------------------------- #
    t_max = 1000
    dt = t_max
    L = 3.0e5  # Domain length
    H = 1.0e4  # Height position of the model top

    columns = 300  # number of columns
    nlayers = 10  # horizontal layers

    # Domain
    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, 'CG', 1)

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
    rexi.solve(U_expl, U_in, t_max)
    u1, p1, b1 = U_expl.subfunctions
    p1 -= p_b
    b1 -= b_b
    rexi_output.write(u1, p1, b1)

    return u1, p1, b1


def test_rexi_linear_boussinesq(tmpdir):

    dirname = str(tmpdir)
    u, p, b = run_rexi_linear_boussinesq(dirname)

    for variable in ['u', 'b', 'p']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        diff_array = new_variable.dat.data - check_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(check_variable.dat.data)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'Incompressible test do not match KGO values'
