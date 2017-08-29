from gusto import *
from firedrake import SpatialCoordinate, \
    as_vector, VectorFunctionSpace, sin, exp, Function, FunctionSpace
import pytest
from math import pi


def setup_model(state, physical_domain, timestepping, field_eqns, time_discretisation):

    advected_fields = []
    if time_discretisation == "ssprk":
        for fname, equation in field_eqns.items():
            advected_fields.append((fname, SSPRK3(state.fields(fname),
                                                  timestepping.dt,
                                                  equation)))
    elif time_discretisation == "implicit_midpoint":
        for fname, equation in field_eqns.items():
            advected_fields.append((fname, ThetaMethod(state.fields(fname),
                                                       timestepping.dt,
                                                       equation)))

    return AdvectionDiffusionModel(state, physical_domain,
                                   timestepping=timestepping,
                                   advected_fields=advected_fields)


def run(model, tmax, fieldlist):

    timestepper = AdvectionTimestepper(model)
    f_dict = timestepper.run(0, tmax, x_end=fieldlist)
    return f_dict


def check_errors(f_dict, f_end, field_eqns, tol):
    f_err = Function(f_end.function_space())
    errors = {}
    for name, field in f_dict.items():
        f_err.assign(f_end - field)
        errors[name] = max(abs(f_err.dat.data.min()), abs(f_err.dat.data.max()))
    for name, err in errors.items():
        if err > tol:
            print("Test fails for field %s with options ibp = %s, continuity = %s" % (name, field_eqns[name].ibp, field_eqns[name].continuity))
    assert(all([err < tol for err in errors.values()]))


@pytest.mark.parametrize("time_discretisation", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("vector", [False, True])
def test_advection_2Dsphere(tmpdir, time_discretisation, vector):

    # set up physical_domain and state
    physical_domain = Sphere(radius=1., ref_level=2)
    dirname = str(tmpdir)
    fieldlist = ["f1", "f2", "f3", "f4"]
    dumplist = fieldlist.append("f_end")
    output = OutputParameters(dirname=dirname, dumplist=dumplist, dumpfreq=15)
    state = AdvectionDiffusionState(physical_domain.mesh,
                                    horizontal_degree=1, family="BDM",
                                    output=output)

    # initial conditions, depending on whether we're advecting a
    # vector or scalar field
    x = SpatialCoordinate(physical_domain.mesh)
    if vector:
        fspace = VectorFunctionSpace(physical_domain.mesh, "DG", 1)
        fexpr = as_vector([exp(-x[2]**2 - x[1]**2), 0., 0.])
    else:
        fspace = state.spaces("DG")
        fexpr = exp(-x[2]**2 - x[1]**2)
    for fname in fieldlist:
        f = state.fields(fname, fspace)
        f.interpolate(fexpr)

    # advecting velocity
    uspace = state.spaces("HDiv")
    u0 = state.fields("u", uspace)
    u0.project(as_vector([-x[1], x[0], 0.0]))

    # timestepping parameters
    timestepping = TimesteppingParameters(dt=pi/3*0.01)

    # equations - we are testing both forms of the advection equation
    # (advective and continuity) and integrating by parts once and
    # twice for each option - this give us 4 combinations.
    eqnlist = []
    eqnlist.append(AdvectionEquation(physical_domain, fspace, uspace))
    eqnlist.append(AdvectionEquation(physical_domain, fspace, uspace, ibp="twice"))
    eqnlist.append(AdvectionEquation(physical_domain, fspace, uspace, equation_form="continuity"))
    eqnlist.append(AdvectionEquation(physical_domain, fspace, uspace, ibp="twice", equation_form="continuity"))

    # setup the model with the above equations and the time_discretisation
    field_eqns = {fname: eqn for (fname, eqn) in zip(fieldlist, eqnlist)}
    model = setup_model(state, physical_domain, timestepping, field_eqns, time_discretisation)

    # run the model
    f_dict = run(model, pi/2., fieldlist)

    # create function, interpolate analytic solution and check errors
    f_end = Function(fspace)
    if vector:
        f_end_expr = as_vector([exp(-x[2]**2 - x[0]**2), 0., 0.])
    else:
        f_end_expr = exp(-x[2]**2 - x[0]**2)
    f_end.interpolate(f_end_expr)
    check_errors(f_dict, f_end, field_eqns, tol=2.5e-2)


@pytest.mark.parametrize("broken", [True, False])
def test_advection_embedded_dg(tmpdir, broken):

    # set up physical_domain and state
    physical_domain = VerticalSlice(H=1., L=1., ncolumns=15, nlayers=15)
    dirname = str(tmpdir)
    fieldlist = ["f1", "f2", "f3", "f4"]
    output = OutputParameters(dirname=dirname, dumplist=fieldlist, dumpfreq=15)
    state = AdvectionDiffusionState(physical_domain.mesh, vertical_degree=1,
                                    horizontal_degree=1, family="CG",
                                    output=output)

    # initial conditions
    x = SpatialCoordinate(physical_domain.mesh)
    fexpr = sin(2*pi*x[0])*sin(2*pi*x[1])
    fspace = state.spaces("HDiv_v")
    for fname in fieldlist:
        f = state.fields(fname, fspace)
        f.interpolate(fexpr)

    # advecting velocity
    uspace = state.spaces("HDiv")
    u0 = state.fields("u", uspace)
    u0.project(as_vector([1.0, 0.0]))

    # timestepping parameters
    timestepping = TimesteppingParameters(dt=0.01)

    # If "broken" the EmbeddedDGAdvection class creates the broken
    # space relating to fspace, if not, we pass in the smallest DG
    # space that fspace is contained in.
    if broken:
        Vdg = None
    else:
        Vdg = state.spaces("DG")

    # equations - we are testing both forms of the advection equation
    # (advective and continuity) and integrating by parts once and
    # twice for each option - this give us 4 combinations.
    eqnlist = []
    eqnlist.append(EmbeddedDGAdvection(physical_domain, fspace, uspace, Vdg=Vdg))
    eqnlist.append(EmbeddedDGAdvection(physical_domain, fspace, uspace, ibp="twice", Vdg=Vdg))
    eqnlist.append(EmbeddedDGAdvection(physical_domain, fspace, uspace, equation_form="continuity", Vdg=Vdg))
    eqnlist.append(EmbeddedDGAdvection(physical_domain, fspace, uspace, ibp="twice", equation_form="continuity", Vdg=Vdg))

    # setup the model with the above equations and the time_discretisation
    field_eqns = {fname: eqn for (fname, eqn) in zip(fieldlist, eqnlist)}
    model = setup_model(state, physical_domain, timestepping, field_eqns, time_discretisation="ssprk")

    # run the model
    f_dict = run(model, 2.5, fieldlist)

    # create function, interpolate analytic solution and check errors
    f_end = Function(fspace)
    f_end_expr = sin(2*pi*(x[0]-0.5))*sin(2*pi*x[1])
    f_end.interpolate(f_end_expr)
    check_errors(f_dict, f_end, field_eqns, tol=7e-2)


@pytest.mark.parametrize("time_discretisation", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("dg_direction", [None, "horizontal"])
def test_advection_supg(tmpdir, time_discretisation, dg_direction):

    # set up physical_domain and state
    physical_domain = VerticalSlice(H=1., L=1., ncolumns=15, nlayers=15)
    dirname = str(tmpdir)
    fieldlist = ["f1", "f2"]
    output = OutputParameters(dirname=dirname, dumplist=fieldlist, dumpfreq=15)
    state = AdvectionDiffusionState(physical_domain.mesh, vertical_degree=1,
                                    horizontal_degree=1, family="CG",
                                    output=output)

    # initial conditions
    x = SpatialCoordinate(physical_domain.mesh)
    fexpr = sin(2*pi*x[0])*sin(2*pi*x[1])

    # dg_direction specifies the direction in which the space is
    # discontinuous, if None then we are fully continous and do not
    # integrate by parts, otherwise we integrate by parts twice and
    # use the HDiv_v space.
    if dg_direction:
        ibp = "twice"
        fspace = state.spaces("HDiv_v")
    else:
        ibp = None
        fspace = FunctionSpace(physical_domain.mesh, "CG", 1)
    for fname in fieldlist:
        f = state.fields(fname, fspace)
        f.interpolate(fexpr)

    # advecting velocity
    uspace = state.spaces("HDiv")
    u0 = state.fields("u", uspace)
    u0.project(as_vector([1.0, 0.0]))

    # timestepping parameters
    timestepping = TimesteppingParameters(dt=0.01)

    # equations - we are testing both forms of the advection equation
    # (advective and continuity) and either integrating by parts twice
    # or not integrating by parts at all, depending on whether we are
    # partially or fully continuous.
    eqnlist = []
    eqnlist.append(SUPGAdvection(physical_domain, fspace, uspace, dt=timestepping.dt, ibp=ibp, supg_params={"dg_direction": dg_direction}))
    eqnlist.append(SUPGAdvection(physical_domain, fspace, uspace, dt=timestepping.dt, ibp=ibp, equation_form="continuity", supg_params={"dg_direction": dg_direction}))

    # setup the model with the above equations and the time_discretisation
    field_eqns = {fname: eqn for (fname, eqn) in zip(fieldlist, eqnlist)}
    model = setup_model(state, physical_domain, timestepping, field_eqns, time_discretisation="ssprk")

    # run the model
    f_dict = run(model, 2.5, fieldlist)

    # create function, interpolate analytic solution and check errors
    f_end = Function(fspace)
    f_end_expr = sin(2*pi*(x[0]-0.5))*sin(2*pi*x[1])
    f_end.interpolate(f_end_expr)
    check_errors(f_dict, f_end, field_eqns, tol=7e-2)
