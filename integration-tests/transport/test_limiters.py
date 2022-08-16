"""
This tests three limiter options for different transport schemes.
A sharp bubble of warm air is generated in a vertical slice and then transported
by a prescribed transport scheme. If the limiter is working, the transport
should have produced no new maxima or minima.
"""

from os import path
from gusto import *
from firedrake import (as_vector, PeriodicIntervalMesh, interval, FiniteElement, pi,
                       SpatialCoordinate, ExtrudedMesh, FunctionSpace, Function,
                       conditional, sqrt, BrokenElement, TensorProductElement)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from netCDF4 import Dataset


def setup_limiters(dirname):

    dt = 0.01
    Ld = 1.
    tmax = 0.2
    rotations = 0.1
    m = PeriodicIntervalMesh(20, Ld)
    mesh = ExtrudedMesh(m, layers=20, layer_height=(Ld/20))
    output = OutputParameters(dirname=dirname, dumpfreq=1, dumplist=['u', 'chemical', 'moisture_higher', 'moisture_lower'])
    parameters = CompressibleParameters()

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters)

    # set up the function spaces:
    Vr = state.spaces("DG1_equispaced")
    Vt = state.spaces("theta", degree=1)
    Vpsi = FunctionSpace(mesh, "CG", 2)

    cell = mesh._base_mesh.ufl_cell().cellname()
    DG0_element = FiniteElement("DG", cell, 0)
    CG1_element = FiniteElement("CG", interval, 1)
    Vt0_element = TensorProductElement(DG0_element, CG1_element)
    Vt0 = FunctionSpace(mesh, Vt0_element)
    Vt0_brok = FunctionSpace(mesh, BrokenElement(Vt0_element))
    VCG1 = FunctionSpace(mesh, "CG", 1)

    # set up the equations:
    chemeqn = AdvectionEquation(state, Vr, "chemical", ufamily="CG", udegree=1)
    moisteqn_higher = AdvectionEquation(state, Vt, "moisture_higher")
    moisteqn_lower = AdvectionEquation(state, Vt0, "moisture_lower")

    x, z = SpatialCoordinate(mesh)

    u = state.fields("u")

    x_lower = 2 * Ld / 5
    x_upper = 3 * Ld / 5
    z_lower = 6 * Ld / 10
    z_upper = 8 * Ld / 10
    bubble_expr_1 = conditional(x > x_lower,
                                conditional(x < x_upper,
                                            conditional(z > z_lower,
                                                        conditional(z < z_upper, 1.0, 0.0),
                                                        0.0),
                                            0.0),
                                0.0)

    bubble_expr_2 = conditional(x > z_lower,
                                conditional(x < z_upper,
                                            conditional(z > x_lower,
                                                        conditional(z < x_upper, 1.0, 0.0),
                                                        0.0),
                                            0.0),
                                0.0)

    chemical = state.fields("chemical")
    moisture_higher = state.fields("moisture_higher")
    moisture_lower = state.fields("moisture_lower")

    chemical.assign(1.0)
    moisture_higher.assign(280.)
    chem_pert_1 = Function(Vr).interpolate(bubble_expr_1)
    chem_pert_2 = Function(Vr).interpolate(bubble_expr_2)
    moist_h_pert_1 = Function(Vt).interpolate(bubble_expr_1)
    moist_h_pert_2 = Function(Vt).interpolate(bubble_expr_2)
    moist_l_pert_1 = Function(Vt0).interpolate(bubble_expr_1)
    moist_l_pert_2 = Function(Vt0).interpolate(bubble_expr_2)

    chemical.assign(chemical + chem_pert_1 + chem_pert_2)
    moisture_higher.assign(moisture_higher + moist_h_pert_1 + moist_h_pert_2)
    moisture_lower.assign(moisture_lower + moist_l_pert_1 + moist_l_pert_2)

    # set up solid body rotation for transport
    # we do this slightly complicated stream function to make the velocity 0 at edges
    # thus we avoid odd effects at boundaries
    xc = Ld / 2
    zc = Ld / 2
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    omega = rotations * 2 * pi / tmax
    r_out = 9 * Ld / 20
    r_in = 2 * Ld / 5
    A = omega * r_in / (2 * (r_in - r_out))
    B = - omega * r_in * r_out / (r_in - r_out)
    C = omega * r_in ** 2 * r_out / (r_in - r_out) / 2
    psi_expr = conditional(r < r_in,
                           omega * r ** 2 / 2,
                           conditional(r < r_out,
                                       A * r ** 2 + B * r + C,
                                       A * r_out ** 2 + B * r_out + C))
    psi = Function(Vpsi).interpolate(psi_expr)

    gradperp = lambda v: as_vector([-v.dx(1), v.dx(0)])
    u.project(gradperp(psi))

    # set up transport schemes
    dg_opts = EmbeddedDGOptions()
    recovered_opts = RecoveredOptions(embedding_space=Vr,
                                      recovered_space=VCG1,
                                      broken_space=Vt0_brok,
                                      boundary_method=Boundary_Method.dynamics)

    transport_schemes = []
    transport_schemes.append((chemeqn, SSPRK3(state, limiter=VertexBasedLimiter(Vr))))
    transport_schemes.append((moisteqn_higher, SSPRK3(state, options=dg_opts, limiter=ThetaLimiter(Vt))))
    transport_schemes.append((moisteqn_lower, SSPRK3(state, options=recovered_opts, limiter=VertexBasedLimiter(Vr))))

    # build time stepper
    stepper = PrescribedTransport(state, transport_schemes)

    return stepper, tmax


def run_limiters(dirname):

    stepper, tmax = setup_limiters(dirname)
    stepper.run(t=0, tmax=tmax)
    return


def test_limiters(tmpdir):

    dirname = str(tmpdir)
    run_limiters(dirname)
    filename = path.join(dirname, "diagnostics.nc")
    data = Dataset(filename, "r")

    chem_data = data.groups["chemical"]
    max_chem = chem_data.variables["max"]
    min_chem = chem_data.variables["min"]

    moist_h_data = data.groups["moisture_higher"]
    max_moist_h = moist_h_data.variables["max"]
    min_moist_h = moist_h_data.variables["min"]

    moist_l_data = data.groups["moisture_lower"]
    max_moist_l = moist_l_data.variables["max"]
    min_moist_l = moist_l_data.variables["min"]

    tolerance = 1e-8

    # check that maxima and minima do not exceed previous maxima and minima
    # however provide a small amount of tolerance
    assert max_chem[-1] <= max_chem[0] + (max_chem[0] - min_chem[0]) * tolerance
    assert min_chem[-1] >= min_chem[0] - (max_chem[0] - min_chem[0]) * tolerance
    assert max_moist_l[-1] <= max_moist_l[0] + (max_moist_l[0] - min_moist_l[0]) * tolerance
    assert min_moist_l[-1] >= min_moist_l[0] - (max_moist_l[0] - min_moist_l[0]) * tolerance
    assert max_moist_h[-1] <= max_moist_h[0] + (max_moist_h[0] - min_moist_h[0]) * tolerance
    assert min_moist_h[-1] >= min_moist_h[0] - (max_moist_h[0] - min_moist_h[0]) * tolerance
