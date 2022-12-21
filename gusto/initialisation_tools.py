"""Tools for computing initial conditions, such as hydrostatic balance."""

from firedrake import MixedFunctionSpace, TrialFunctions, TestFunctions, \
    TestFunction, TrialFunction, SpatialCoordinate, \
    FacetNormal, inner, div, dx, ds_b, ds_t, DirichletBC, \
    Function, Constant, \
    LinearVariationalProblem, LinearVariationalSolver, \
    NonlinearVariationalProblem, NonlinearVariationalSolver, split, solve, \
    sin, cos, sqrt, asin, atan_2, as_vector, Min, Max, FunctionSpace, \
    errornorm, zero
from gusto import thermodynamics
from gusto.configuration import logger
from gusto.recovery import Recoverer, BoundaryMethod


__all__ = ["latlon_coords", "sphere_to_cartesian", "incompressible_hydrostatic_balance",
           "compressible_hydrostatic_balance", "remove_initial_w",
           "saturated_hydrostatic_balance", "unsaturated_hydrostatic_balance"]


# TODO: maybe coordinate transforms could go elsewhere
def latlon_coords(mesh):
    """
    Gets expressions for the latitude and longitude fields.

    Args:
        mesh (:class:`Mesh`): the model's mesh.

    Returns:
        tuple of :class:`ufl.Expr`: expressions for the latitude and longitude
            fields, in radians.
    """
    x0, y0, z0 = SpatialCoordinate(mesh)
    unsafe = z0/sqrt(x0*x0 + y0*y0 + z0*z0)
    safe = Min(Max(unsafe, -1.0), 1.0)  # avoid silly roundoff errors
    theta = asin(safe)  # latitude
    lamda = atan_2(y0, x0)  # longitude
    return theta, lamda


def sphere_to_cartesian(mesh, u_zonal, u_merid):
    """
    Convert the horizontal spherical-polar components of a vector into
    geocentric Cartesian components.

    Args:
        mesh (:class:`Mesh`): _description_
        u_zonal (:class:`ufl.Expr`): the zonal component of the vector.
        u_merid (:class:`ufl.Expr`): the meridional component of the vector.

    Returns:
        _type_: _description_
    """
    theta, lamda = latlon_coords(mesh)

    cartesian_u_expr = -u_zonal*sin(lamda) - u_merid*sin(theta)*cos(lamda)
    cartesian_v_expr = u_zonal*cos(lamda) - u_merid*sin(theta)*sin(lamda)
    cartesian_w_expr = u_merid*cos(theta)

    return as_vector((cartesian_u_expr, cartesian_v_expr, cartesian_w_expr))


def incompressible_hydrostatic_balance(equation, b0, p0, top=False, params=None):
    """
    Gives a pressure field in hydrostatic-balance for the Incompressible eqns.

    Generates the hydrostatically-balanced pressure field for the incompressible
    Boussinesq equations, given some buoyancy field and a boundary condition.
    This is solved as a mixed problem for the vertical velocity and the pressure
    with zero flow enforced at one of the boundaries.

    Args:
        equation (:class:`PrognosticEquation`): the model's equation object.
        b0 (:class:`ufl.Expr`): the input buoyancy field.
        p0 (:class:`Function`): the pressure to be returned.
        top (bool, optional): whether the no-flow boundary condition is enforced
            on the top boundary or the bottom. True denotes the top. Defaults to
            False.
        params (dict, optional): dictionary of parameters to be passed to the
            solver. Defaults to None.
    """

    # get F
    domain = equation.domain
    Vu = domain.spaces("HDiv")
    Vv = FunctionSpace(equation.domain.mesh, Vu.ufl_element()._elements[-1])
    v = TrialFunction(Vv)
    w = TestFunction(Vv)

    if top:
        bstring = "top"
    else:
        bstring = "bottom"

    bcs = [DirichletBC(Vv, 0.0, bstring)]

    a = inner(w, v)*dx
    L = inner(equation.domain.k, w)*b0*dx
    F = Function(Vv)

    solve(a == L, F, bcs=bcs)

    # define mixed function space
    VDG = domain.spaces("DG")
    WV = (Vv)*(VDG)

    # get pprime
    v, pprime = TrialFunctions(WV)
    w, phi = TestFunctions(WV)

    bcs = [DirichletBC(WV[0], zero(), bstring)]

    a = (
        inner(w, v) + div(w)*pprime + div(v)*phi
    )*dx
    L = phi*div(F)*dx
    w1 = Function(WV)

    if params is None:
        params = {'ksp_type': 'gmres',
                  'pc_type': 'fieldsplit',
                  'pc_fieldsplit_type': 'schur',
                  'pc_fieldsplit_schur_fact_type': 'full',
                  'pc_fieldsplit_schur_precondition': 'selfp',
                  'fieldsplit_1_ksp_type': 'preonly',
                  'fieldsplit_1_pc_type': 'gamg',
                  'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                  'fieldsplit_1_mg_levels_sub_pc_type': 'ilu',
                  'fieldsplit_0_ksp_type': 'richardson',
                  'fieldsplit_0_ksp_max_it': 4,
                  'ksp_atol': 1.e-08,
                  'ksp_rtol': 1.e-08}

    solve(a == L, w1, bcs=bcs, solver_parameters=params)

    v, pprime = w1.split()
    p0.project(pprime)


def compressible_hydrostatic_balance(equation, theta0, rho0, exner0=None,
                                     top=False, exner_boundary=Constant(1.0),
                                     mr_t=None,
                                     solve_for_rho=False,
                                     params=None):
    """
    Computes hydrostatic balance for the compressible Euler equations.

    Compute a hydrostatically balanced density or pressure given a potential
    temperature profile. This solves a mixed finite element problem for the
    pressure and the vertical velocity, with an option to subsequently solve for
    the density. By default, this uses a vertically-oriented hybridization
    procedure for solving the resulting discrete systems.

    Args:
        equation (:class:`PrognosticEquation`): the model's equation object.
        theta0 (:class:`ufl.Expr`): the input (dry) potential temperature field.
        rho0 (:class:`Function`): the hydrostatically-balanced density to be
            found.
        exner0 (:class:`Function`, optional): the hydrostatically-balanced Exner
            pressure field. If provided, then the Exner pressure computed as
            part of this routine will be stored in this function. Defaults to
            None.
        top (bool, optional): whether the pressure boundary condition is defined
            on the top boundary or the bottom. True denotes the top. Defaults to
            False.
        exner_boundary (:class:`ufl.Expr`, optional): the Exner pressure field
            on the boundary defining the boundary condition. Defaults to
            `Constant(1.0)`.
        mr_t (:class:`ufl.Expr`, optional): the total water mixing ratio field.
            Defaults to None.
        solve_for_rho (bool, optional): whether to perform a final solve for the
            density field. If false, interpolate rho from the Exner pressure
            using the equation of state. Defaults to False.
        params (dict, optional): dictionary of parameters to be passed to the
            solver. Defaults to None.
    """

    # Calculate hydrostatic Pi
    domain = equation.domain
    parameters = equation.parameters
    VDG = domain.spaces("DG")
    Vu = domain.spaces("HDiv")
    Vv = FunctionSpace(equation.domain.mesh, Vu.ufl_element()._elements[-1])
    W = MixedFunctionSpace((Vv, VDG))
    v, exner = TrialFunctions(W)
    dv, dexner = TestFunctions(W)

    n = FacetNormal(equation.domain.mesh)

    cp = parameters.cp

    # add effect of density of water upon theta
    theta = theta0

    if mr_t is not None:
        theta = theta0 / (1 + mr_t)

    alhs = (
        (cp*inner(v, dv) - cp*div(dv*theta)*exner)*dx
        + dexner*div(theta*v)*dx
    )

    if top:
        bmeasure = ds_t
        bstring = "bottom"
    else:
        bmeasure = ds_b
        bstring = "top"

    arhs = -cp*inner(dv, n)*theta*exner_boundary*bmeasure

    # Possibly make g vary with spatial coordinates?
    g = parameters.g

    arhs -= g*inner(dv, equation.domain.k)*dx

    bcs = [DirichletBC(W.sub(0), zero(), bstring)]

    w = Function(W)
    exner_problem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)

    if params is None:
        params = {'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'mat_type': 'matfree',
                  'pc_python_type': 'gusto.VerticalHybridizationPC',
                  # Vertical trace system is only coupled vertically in columns
                  # block ILU is a direct solver!
                  'vert_hybridization': {'ksp_type': 'preonly',
                                         'pc_type': 'bjacobi',
                                         'sub_pc_type': 'ilu'}}

    exner_solver = LinearVariationalSolver(exner_problem,
                                           solver_parameters=params,
                                           options_prefix="exner_solver")

    exner_solver.solve()
    v, exner = w.split()
    if exner0 is not None:
        exner0.assign(exner)

    if solve_for_rho:
        w1 = Function(W)
        v, rho = w1.split()
        rho.interpolate(thermodynamics.rho(parameters, theta0, exner))
        v, rho = split(w1)
        dv, dexner = TestFunctions(W)
        exner = thermodynamics.exner_pressure(parameters, rho, theta0)
        F = (
            (cp*inner(v, dv) - cp*div(dv*theta)*exner)*dx
            + dexner*div(theta0*v)*dx
            + cp*inner(dv, n)*theta*exner_boundary*bmeasure
        )
        F += g*inner(dv, equation.domain.k)*dx
        rhoproblem = NonlinearVariationalProblem(F, w1, bcs=bcs)
        rhosolver = NonlinearVariationalSolver(rhoproblem, solver_parameters=params,
                                               options_prefix="rhosolver")
        rhosolver.solve()
        v, rho_ = w1.split()
        rho0.assign(rho_)
    else:
        rho0.interpolate(thermodynamics.rho(parameters, theta0, exner))


def remove_initial_w(u):
    """
    Removes the vertical component of a velocity field.

    Args:
        u (:class:`Function`): the velocity field to be altered.
    """
    Vu = u.function_space()
    Vv = FunctionSpace(Vu._ufl_domain, Vu.ufl_element()._elements[-1])
    bc = DirichletBC(Vu[0], 0.0, "bottom")
    bc.apply(u)
    uv = Function(Vv).project(u)
    ustar = Function(u.function_space()).project(uv)
    uin = Function(u.function_space()).assign(u - ustar)
    u.assign(uin)


def saturated_hydrostatic_balance(equation, state_fields, theta_e, mr_t,
                                  exner0=None, top=False,
                                  exner_boundary=Constant(1.0),
                                  max_outer_solve_count=40,
                                  max_theta_solve_count=5,
                                  max_inner_solve_count=3):
    """
    Computes hydrostatic balance for a moist saturated, compressible atmosphere.

    Given a wet equivalent potential temperature, theta_e, and the total
    moisture content, mr_t, compute a hydrostatically balanced virtual dry
    potential temperature, dry density and water vapour profile.

    The general strategy is to split up the solving into two steps:
    1) finding rho to balance the theta profile
    2) finding theta_vd and r_v to get back theta_e and saturation, using a fixed
        point iteration.
    We iteratively solve these steps until we (hopefully)
    converge to a solution.

    Args:
        equation (:class:`PrognosticEquation`): the model's equation object.
        state_fields (:class:`StateFields`): the model's field container.
        theta_e (:class:`ufl.Expr`): expression for the desired wet equivalent
            potential temperature field.
        mr_t (:class:`ufl.Expr`): expression for the total moisture content.
        exner0 (:class:`Function`, optional): the hydrostatically-balanced Exner
            pressure field. If provided, then the Exner pressure computed as
            part of this routine will be stored in this function. Defaults to
            None.
        top (bool, optional): whether the pressure boundary condition is defined
            on the top boundary or the bottom. True denotes the top. Defaults to
            False.
        exner_boundary (:class:`ufl.Expr`, optional): the Exner pressure field
            on the boundary defining the boundary condition. Defaults to
            `Constant(1.0)`.
        max_outer_solve_count (int, optional): maximum number of outer solves
            to perform. Defaults to 40.
        max_theta_solve_count (int, optional): maximum number of solves for the
            theta_vd field, per outer loop. Defaults to 5.
        max_inner_solve_count (int, optional): maximum number of inner solves,
            for the moisture fields, per theta solve. Defaults to 3.

    Raises:
        RuntimeError: if the prognostic fields have not converged to give the
            specified profile to the desired tolerance, within the maximum
            number of iterations.
    """

    theta0 = state_fields('theta')
    rho0 = state_fields('rho')
    mr_v0 = state_fields('water_vapour')

    # Calculate hydrostatic exner pressure
    domain = equation.domain
    parameters = equation.parameters
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    VDG = domain.spaces("DG")
    if any(deg > 2 for deg in VDG.ufl_element().degree()):
        logger.warning("default quadrature degree most likely not sufficient for this degree element")

    theta0.interpolate(theta_e)
    mr_v0.interpolate(mr_t)

    v_deg = Vr.ufl_element().degree()[1]
    if v_deg == 0:
        boundary_method = BoundaryMethod.extruded
    else:
        boundary_method = None
    rho_h = Function(Vr)
    rho_averaged = Function(Vt)
    rho_recoverer = Recoverer(rho0, rho_averaged, boundary_method=boundary_method)
    w_h = Function(Vt)
    theta_h = Function(Vt)
    theta_e_test = Function(Vt)
    delta = 0.8

    # expressions for finding theta0 and mr_v0 from theta_e and mr_t
    exner = thermodynamics.exner_pressure(parameters, rho_averaged, theta0)
    p = thermodynamics.p(parameters, exner)
    T = thermodynamics.T(parameters, theta0, exner, mr_v0)
    r_v_expr = thermodynamics.r_sat(parameters, T, p)
    theta_e_expr = thermodynamics.theta_e(parameters, T, p, mr_v0, mr_t)

    for i in range(max_outer_solve_count):
        # solve for rho with theta_vd and w_v guesses
        compressible_hydrostatic_balance(equation, theta0, rho_h, top=top,
                                         exner_boundary=exner_boundary, mr_t=mr_t,
                                         solve_for_rho=True)

        # damp solution
        rho0.assign(rho0 * (1 - delta) + delta * rho_h)

        theta_e_test.interpolate(theta_e_expr)
        if errornorm(theta_e_test, theta_e) < 1e-8:
            break

        # calculate averaged rho
        rho_recoverer.project()

        # now solve for r_v
        for j in range(max_theta_solve_count):
            theta_h.interpolate(theta_e / theta_e_expr * theta0)
            theta0.assign(theta0 * (1 - delta) + delta * theta_h)

            # break when close enough
            if errornorm(theta_e_test, theta_e) < 1e-6:
                break
            for k in range(max_inner_solve_count):
                w_h.interpolate(r_v_expr)
                mr_v0.assign(mr_v0 * (1 - delta) + delta * w_h)

                # break when close enough
                theta_e_test.interpolate(theta_e_expr)
                if errornorm(theta_e_test, theta_e) < 1e-6:
                    break

        if i == max_outer_solve_count:
            raise RuntimeError('Hydrostatic balance solve has not converged within %i' % i, 'iterations')

    if exner0 is not None:
        exner = thermodynamics.exner(parameters, rho0, theta0)
        exner0.interpolate(exner)

    # do one extra solve for rho
    compressible_hydrostatic_balance(equation, theta0, rho0, top=top,
                                     exner_boundary=exner_boundary,
                                     mr_t=mr_t, solve_for_rho=True)


def unsaturated_hydrostatic_balance(equation, state_fields, theta_d, H,
                                    exner0=None, top=False,
                                    exner_boundary=Constant(1.0),
                                    max_outer_solve_count=40,
                                    max_inner_solve_count=20):
    """
    Computes hydrostatic bal. for a moist unsaturated, compressible atmosphere.

    Given vertical profiles for dry potential temperature and relative humidity,
    computes hydrostatically balanced virtual dry potential temperature, dry
    density and water vapour profiles.

    The general strategy is to split up the solving into two steps:
    1) finding rho to balance the theta profile
    2) finding theta_v and r_v to get back theta_d and H, using a fixed-point
       iteration.
    These steps are iterated until we (hopefully) converge to a solution.

    Args:
        equation (:class:`PrognosticEquation`): the model's equation object.
        state_fields (:class:`StateFields`): the model's field container.
        theta_d (:class:`ufl.Expr`): the specified dry potential temperature
            field.
        H (:class:`ufl.Expr`): the specified relative humidity field.
        exner0 (:class:`Function`, optional): the hydrostatically-balanced Exner
            pressure field. If provided, then the Exner pressure computed as
            part of this routine will be stored in this function. Defaults to
            None.
        top (bool, optional): whether the pressure boundary condition is defined
            on the top boundary or the bottom. True denotes the top. Defaults to
            False.
        exner_boundary (:class:`ufl.Expr`, optional): the Exner pressure field
            on the boundary defining the boundary condition. Defaults to
            `Constant(1.0)`.
        max_outer_solve_count (int, optional): maximum number of outer solves
            to perform. Defaults to 40.
        max_inner_solve_count (int, optional): maximum number of inner solves,
            for the moisture fields, per outer solve. Defaults to 20.

    Raises:
        RuntimeError: if the prognostic fields have not converged to give the
            specified profile to the desired tolerance, within the maximum
            number of iterations.
    """

    theta0 = state_fields('theta')
    rho0 = state_fields('rho')
    mr_v0 = state_fields('water_vapour')

    # Calculate hydrostatic exner pressure
    domain = equation.domain
    parameters = equation.parameters
    Vt = theta0.function_space()
    Vr = rho0.function_space()
    R_d = parameters.R_d
    R_v = parameters.R_v
    epsilon = R_d / R_v

    VDG = domain.spaces("DG")
    if any(deg > 2 for deg in VDG.ufl_element().degree()):
        logger.warning("default quadrature degree most likely not sufficient for this degree element")

    # apply first guesses
    theta0.assign(theta_d * 1.01)
    mr_v0.assign(0.01)

    v_deg = Vr.ufl_element().degree()[1]
    if v_deg == 0:
        method = BoundaryMethod.extruded
    else:
        method = None
    rho_h = Function(Vr)
    rho_averaged = Function(Vt)
    rho_recoverer = Recoverer(rho0, rho_averaged, boundary_method=method)
    w_h = Function(Vt)
    delta = 1.0

    # make expressions for determining mr_v0
    exner = thermodynamics.exner_pressure(parameters, rho_averaged, theta0)
    p = thermodynamics.p(parameters, exner)
    T = thermodynamics.T(parameters, theta0, exner, mr_v0)
    r_v_expr = thermodynamics.r_v(parameters, H, T, p)

    # make expressions to evaluate residual
    exner_ev = thermodynamics.exner_pressure(parameters, rho_averaged, theta0)
    p_ev = thermodynamics.p(parameters, exner_ev)
    T_ev = thermodynamics.T(parameters, theta0, exner_ev, mr_v0)
    RH_ev = thermodynamics.RH(parameters, mr_v0, T_ev, p_ev)
    RH = Function(Vt)

    for i in range(max_outer_solve_count):
        # solve for rho with theta_vd and w_v guesses
        compressible_hydrostatic_balance(equation, theta0, rho_h, top=top,
                                         exner_boundary=exner_boundary, mr_t=mr_v0,
                                         solve_for_rho=True)

        # damp solution
        rho0.assign(rho0 * (1 - delta) + delta * rho_h)

        # calculate averaged rho
        rho_recoverer.project()

        RH.interpolate(RH_ev)
        if errornorm(RH, H) < 1e-10:
            break

        # now solve for r_v
        for j in range(max_inner_solve_count):
            w_h.interpolate(r_v_expr)
            mr_v0.assign(mr_v0 * (1 - delta) + delta * w_h)

            # compute theta_vd
            theta0.interpolate(theta_d * (1 + mr_v0 / epsilon))

            # test quality of solution by re-evaluating expression
            RH.interpolate(RH_ev)
            if errornorm(RH, H) < 1e-10:
                break

        if i == max_outer_solve_count:
            raise RuntimeError('Hydrostatic balance solve has not converged within %i' % i, 'iterations')

    if exner0 is not None:
        exner = thermodynamics.exner_pressure(parameters, rho0, theta0)
        exner0.interpolate(exner)

    # do one extra solve for rho
    compressible_hydrostatic_balance(equation, theta0, rho0, top=top,
                                     exner_boundary=exner_boundary,
                                     mr_t=mr_v0, solve_for_rho=True)
