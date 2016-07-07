from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    as_vector, VectorFunctionSpace
from math import pi


def setup_DGadvection(dirname):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)
    R = 1.
    dt = pi/3*0.01

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements,
                                 degree=3)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    fieldlist = ['u','f_cont']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/DGadvection", dumplist=["f_cont", "f_adv", "fvec_conv", "fvec_adv"])

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = Function(state.V[0])
    x = SpatialCoordinate(mesh)
    uexpr = as_vector([-x[1], x[0], 0.0])
    u0.project(uexpr)

    VectorDGSpace = VectorFunctionSpace(mesh, "DG", 1)
    fvec_expr = Expression(("exp(-pow(x[2],2) - pow(x[1],2))", "0.0", "0.0"))
    fvec_cont = Function(VectorDGSpace, name="fvec_cont").interpolate(fvec_expr)
    fvec_adv = Function(VectorDGSpace, name="fvec_adv").interpolate(fvec_expr)
    fvec_end_expr = Expression(("exp(-pow(x[2],2) - pow(x[0],2))","0","0"))
    fvec_end = Function(VectorDGSpace).interpolate(fvec_end_expr)

    fexpr = Expression("exp(-pow(x[2],2) - pow(x[1],2))")
    f_cont = Function(state.V[1], name="f_cont").interpolate(fexpr)
    f_adv = Function(state.V[1], name="f_adv").interpolate(fexpr)
    f_end_expr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")
    f_end = Function(state.V[1]).interpolate(f_end_expr)

    state.initialise([u0, f_cont, f_adv, fvec_cont, fvec_adv])
    state.field_dict["fvec_cont"] = fvec_cont
    state.field_dict["fvec_adv"] = fvec_adv
    state.field_dict["f_adv"] = f_adv

    advection_dict = {}
    advection_dict["f_cont"] = DGAdvection(state, f_cont.function_space(), continuity=True)
    advection_dict["f_adv"] = DGAdvection(state, f_adv.function_space(), continuity=False)
    advection_dict["fvec_cont"] = DGAdvection(state, fvec_cont.function_space(), continuity=True)
    advection_dict["fvec_adv"] = DGAdvection(state, fvec_adv.function_space(), continuity=False)
    stepper = AdvectionTimestepper(state, advection_dict)
    return stepper, fvec_end, f_end


def run(dirname):

    stepper, fvec_end, f_end = setup_DGadvection(dirname)

    tmax = pi/2.

    field_dict = stepper.run(t=0, tmax=tmax, x_end=["f_cont", "f_adv", "fvec_cont", "fvec_adv"])

    return field_dict, fvec_end, f_end


def test_dgadvection(tmpdir):

    dirname = str(tmpdir)
    field_dict, fvec_end, f_end = run(dirname)
    print field_dict.keys()
    fvec_cont_err = Function(fvec_end.function_space()).assign(field_dict["fvec_cont"] - fvec_end)
    assert(abs(fvec_cont_err.dat.data.max()) < 2.5e-2)
    fvec_adv_err = Function(fvec_end.function_space()).assign(field_dict["fvec_adv"] - fvec_end)
    assert(abs(fvec_adv_err.dat.data.max()) < 2.5e-2)
    f_cont_err = Function(f_end.function_space()).assign(field_dict["f_cont"] - f_end)
    assert(abs(f_cont_err.dat.data.max()) < 2.5e-2)
    f_adv_err = Function(f_end.function_space()).assign(field_dict["f_adv"] - f_end)
    assert(abs(f_adv_err.dat.data.max()) < 2.5e-2)
