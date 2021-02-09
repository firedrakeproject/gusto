"""
A module containing some tools for computing initial conditions, such
as balanced initial conditions.
"""

from firedrake import MixedFunctionSpace, TrialFunctions, TestFunctions, \
    TestFunction, TrialFunction, SpatialCoordinate, \
    FacetNormal, inner, div, dx, ds_b, ds_t, ds_tb, DirichletBC, \
    Function, Constant, assemble, \
    LinearVariationalProblem, LinearVariationalSolver, \
    NonlinearVariationalProblem, NonlinearVariationalSolver, split, solve, \
    sin, cos, sqrt, asin, atan_2, as_vector, Min, Max, FunctionSpace, \
    BrokenElement, errornorm, zero
from gusto import thermodynamics
from gusto.configuration import logger
from gusto.recovery import Recoverer, Boundary_Method


__all__ = ["latlon_coords", "sphere_to_cartesian", "incompressible_hydrostatic_balance", "compressible_hydrostatic_balance", "remove_initial_w", "eady_initial_v", "compressible_eady_initial_v", "calculate_Pi0", "saturated_hydrostatic_balance", "unsaturated_hydrostatic_balance"]


def latlon_coords(mesh):
    x0, y0, z0 = SpatialCoordinate(mesh)
    unsafe = z0/sqrt(x0*x0 + y0*y0 + z0*z0)
    safe = Min(Max(unsafe, -1.0), 1.0)  # avoid silly roundoff errors
    theta = asin(safe)  # latitude
    lamda = atan_2(y0, x0)  # longitude
    return theta, lamda


def sphere_to_cartesian(mesh, u_zonal, u_merid):
    theta, lamda = latlon_coords(mesh)

    cartesian_u_expr = -u_zonal*sin(lamda) - u_merid*sin(theta)*cos(lamda)
    cartesian_v_expr = u_zonal*cos(lamda) - u_merid*sin(theta)*sin(lamda)
    cartesian_w_expr = u_merid*cos(theta)

    return as_vector((cartesian_u_expr, cartesian_v_expr, cartesian_w_expr))


def incompressible_hydrostatic_balance(state, b0, p0, top=False, params=None):

    # get F
    Vv = state.spaces("Vv")
    v = TrialFunction(Vv)
    w = TestFunction(Vv)

    if top:
        bstring = "top"
    else:
        bstring = "bottom"

    bcs = [DirichletBC(Vv, 0.0, bstring)]

    a = inner(w, v)*dx
    L = inner(state.k, w)*b0*dx
    F = Function(Vv)

    solve(a == L, F, bcs=bcs)

    # define mixed function space
    VDG = state.spaces("DG")
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


def compressible_hydrostatic_balance(state, theta0, rho0, pi0=None,
                                     top=False, pi_boundary=Constant(1.0),
                                     water_t=None,
                                     solve_for_rho=False,
                                     params=None):
    """
    Compute a hydrostatically balanced density given a potential temperature
    profile. By default, this uses a vertically-oriented hybridization
    procedure for solving the resulting discrete systems.

    :arg state: The :class:`State` object.
    :arg theta0: :class:`.Function`containing the potential temperature.
    :arg rho0: :class:`.Function` to write the initial density into.
    :arg top: If True, set a boundary condition at the top. Otherwise, set
    it at the bottom.
    :arg pi_boundary: a field or expression to use as boundary data for pi on
    the top or bottom as specified.
    :arg water_t: the initial total water mixing ratio field.
    """

    # Calculate hydrostatic Pi
    VDG = state.spaces("DG")
    Vv = state.spaces("Vv")
    W = MixedFunctionSpace((Vv, VDG))
    v, pi = TrialFunctions(W)
    dv, dpi = TestFunctions(W)

    n = FacetNormal(state.mesh)

    cp = state.parameters.cp

    # add effect of density of water upon theta
    theta = theta0

    if water_t is not None:
        theta = theta0 / (1 + water_t)

    alhs = (
        (cp*inner(v, dv) - cp*div(dv*theta)*pi)*dx
        + dpi*div(theta*v)*dx
    )

    if top:
        bmeasure = ds_t
        bstring = "bottom"
    else:
        bmeasure = ds_b
        bstring = "top"

    arhs = -cp*inner(dv, n)*theta*pi_boundary*bmeasure

    # Possibly make g vary with spatial coordinates?
    g = state.parameters.g

    arhs -= g*inner(dv, state.k)*dx

    bcs = [DirichletBC(W.sub(0), zero(), bstring)]

    w = Function(W)
    PiProblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)

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

    PiSolver = LinearVariationalSolver(PiProblem,
                                       solver_parameters=params,
                                       options_prefix="pisolver")

    PiSolver.solve()
    v, Pi = w.split()
    if pi0 is not None:
        pi0.assign(Pi)

    if solve_for_rho:
        w1 = Function(W)
        v, rho = w1.split()
        rho.interpolate(thermodynamics.rho(state.parameters, theta0, Pi))
        v, rho = split(w1)
        dv, dpi = TestFunctions(W)
        pi = thermodynamics.pi(state.parameters, rho, theta0)
        F = (
            (cp*inner(v, dv) - cp*div(dv*theta)*pi)*dx
            + dpi*div(theta0*v)*dx
            + cp*inner(dv, n)*theta*pi_boundary*bmeasure
        )
        F += g*inner(dv, state.k)*dx
        rhoproblem = NonlinearVariationalProblem(F, w1, bcs=bcs)
        rhosolver = NonlinearVariationalSolver(rhoproblem, solver_parameters=params,
                                               options_prefix="rhosolver")
        rhosolver.solve()
        v, rho_ = w1.split()
        rho0.assign(rho_)
    else:
        rho0.interpolate(thermodynamics.rho(state.parameters, theta0, Pi))


def remove_initial_w(u, Vv):
    bc = DirichletBC(u.function_space()[0], 0.0, "bottom")
    bc.apply(u)
    uv = Function(Vv).project(u)
    ustar = Function(u.function_space()).project(uv)
    uin = Function(u.function_space()).assign(u - ustar)
    u.assign(uin)


def eady_initial_v(state, p0, v):
    f = state.parameters.f
    x, y, z = SpatialCoordinate(state.mesh)

    # get pressure gradient
    Vu = state.spaces("HDiv")
    g = TrialFunction(Vu)
    wg = TestFunction(Vu)

    n = FacetNormal(state.mesh)

    a = inner(wg, g)*dx
    L = -div(wg)*p0*dx + inner(wg, n)*p0*ds_tb
    pgrad = Function(Vu)
    solve(a == L, pgrad)

    # get initial v
    Vp = p0.function_space()
    phi = TestFunction(Vp)
    m = TrialFunction(Vp)

    a = f*phi*m*dx
    L = phi*pgrad[0]*dx
    solve(a == L, v)

    return v


def compressible_eady_initial_v(state, theta0, rho0, v):
    f = state.parameters.f
    cp = state.parameters.cp

    # exner function
    Vr = rho0.function_space()
    Pi = Function(Vr).interpolate(thermodynamics.pi(state.parameters, rho0, theta0))

    # get Pi gradient
    Vu = state.spaces("HDiv")
    g = TrialFunction(Vu)
    wg = TestFunction(Vu)

    n = FacetNormal(state.mesh)

    a = inner(wg, g)*dx
    L = -div(wg)*Pi*dx + inner(wg, n)*Pi*ds_tb
    pgrad = Function(Vu)
    solve(a == L, pgrad)

    # get initial v
    m = TrialFunction(Vr)
    phi = TestFunction(Vr)

    a = phi*f*m*dx
    L = phi*cp*theta0*pgrad[0]*dx
    solve(a == L, v)

    return v


def calculate_Pi0(state, theta0, rho0):
    # exner function
    Vr = rho0.function_space()
    Pi = Function(Vr).interpolate(thermodynamics.pi(state.parameters, rho0, theta0))
    Pi0 = assemble(Pi*dx)/assemble(Constant(1)*dx(domain=state.mesh))

    return Pi0


def saturated_hydrostatic_balance(state, theta_e, water_t, pi0=None,
                                  top=False, pi_boundary=Constant(1.0),
                                  max_outer_solve_count=40,
                                  max_theta_solve_count=5,
                                  max_inner_solve_count=3):
    """
    Given a wet equivalent potential temperature, theta_e, and the total moisture
    content, water_t, compute a hydrostatically balance virtual potential temperature,
    dry density and water vapour profile.

    The general strategy is to split up the solving into two steps:
    1) finding rho to balance the theta profile
    2) finding theta_v and r_v to get back theta_e and saturation
    We iteratively solve these steps until we (hopefully)
    converge to a solution.

    :arg state: The :class:`State` object.
    :arg theta_e: The initial wet equivalent potential temperature profile.
    :arg water_t: The total water pseudo-mixing ratio profile.
    :arg pi0: Optional function to put exner pressure into.
    :arg top: If True, set a boundary condition at the top, otherwise
              it will be at the bottom.
    :arg pi_boundary: The value of pi on the specified boundary.
    :arg max_outer_solve_count: Max number of outer iterations for balance solver.
    :arg max_theta_solve_count: Max number of iterations for theta solver (middle part of solve).
    :arg max_inner_solve_count: Max number of iterations on the inner most
                                loop for the water vapour solver.
    """

    theta0 = state.fields('theta')
    rho0 = state.fields('rho')
    water_v0 = state.fields('water_v')

    # Calculate hydrostatic Pi
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    VDG = state.spaces("DG")
    if any(deg > 2 for deg in VDG.ufl_element().degree()):
        logger.warning("default quadrature degree most likely not sufficient for this degree element")

    theta0.interpolate(theta_e)
    water_v0.interpolate(water_t)

    if state.vertical_degree == 0:
        boundary_method = Boundary_Method.physics
    else:
        boundary_method = None
    rho_h = Function(Vr)
    Vt_broken = FunctionSpace(state.mesh, BrokenElement(Vt.ufl_element()))
    rho_averaged = Function(Vt)
    rho_recoverer = Recoverer(rho0, rho_averaged, VDG=Vt_broken, boundary_method=boundary_method)
    w_h = Function(Vt)
    theta_h = Function(Vt)
    theta_e_test = Function(Vt)
    delta = 0.8

    # expressions for finding theta0 and water_v0 from theta_e and water_t
    pie = thermodynamics.pi(state.parameters, rho_averaged, theta0)
    p = thermodynamics.p(state.parameters, pie)
    T = thermodynamics.T(state.parameters, theta0, pie, water_v0)
    r_v_expr = thermodynamics.r_sat(state.parameters, T, p)
    theta_e_expr = thermodynamics.theta_e(state.parameters, T, p, water_v0, water_t)

    for i in range(max_outer_solve_count):
        # solve for rho with theta_vd and w_v guesses
        compressible_hydrostatic_balance(state, theta0, rho_h, top=top,
                                         pi_boundary=pi_boundary, water_t=water_t,
                                         solve_for_rho=True)

        # damp solution
        rho0.assign(rho0 * (1 - delta) + delta * rho_h)

        theta_e_test.assign(theta_e_expr)
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
                water_v0.assign(water_v0 * (1 - delta) + delta * w_h)

                # break when close enough
                theta_e_test.assign(theta_e_expr)
                if errornorm(theta_e_test, theta_e) < 1e-6:
                    break

        if i == max_outer_solve_count:
            raise RuntimeError('Hydrostatic balance solve has not converged within %i' % i, 'iterations')

    if pi0 is not None:
        pie = thermodynamics.pi(state.parameters, rho0, theta0)
        pi0.interpolate(pie)

    # do one extra solve for rho
    compressible_hydrostatic_balance(state, theta0, rho0, top=top,
                                     pi_boundary=pi_boundary,
                                     water_t=water_t, solve_for_rho=True)


def unsaturated_hydrostatic_balance(state, theta_d, H, pi0=None,
                                    top=False, pi_boundary=Constant(1.0),
                                    max_outer_solve_count=40,
                                    max_inner_solve_count=20):
    """
    Given vertical profiles for dry potential temperature
    and relative humidity compute hydrostatically balanced
    virtual potential temperature, dry density and water vapour profiles.

    The general strategy is to split up the solving into two steps:
    1) finding rho to balance the theta profile
    2) finding theta_v and r_v to get back theta_d and H
    We iteratively solve these steps until we (hopefully)
    converge to a solution.

    :arg state: The :class:`State` object.
    :arg theta_d: The initial dry potential temperature profile.
    :arg H: The relative humidity profile.
    :arg pi0: Optional function to put exner pressure into.
    :arg top: If True, set a boundary condition at the top, otherwise
              it will be at the bottom.
    :arg pi_boundary: The value of pi on the specified boundary.
    :arg max_outer_solve_count: Max number of iterations for outer loop of balance solver.
    :arg max_inner_solve_count: Max number of iterations for inner loop of balanace solver.
    """

    theta0 = state.fields('theta')
    rho0 = state.fields('rho')
    water_v0 = state.fields('water_v')

    # Calculate hydrostatic Pi
    Vt = theta0.function_space()
    Vr = rho0.function_space()
    R_d = state.parameters.R_d
    R_v = state.parameters.R_v
    epsilon = R_d / R_v

    VDG = state.spaces("DG")
    if any(deg > 2 for deg in VDG.ufl_element().degree()):
        logger.warning("default quadrature degree most likely not sufficient for this degree element")

    # apply first guesses
    theta0.assign(theta_d * 1.01)
    water_v0.assign(0.01)

    if state.vertical_degree == 0:
        method = Boundary_Method.physics
    else:
        method = None
    rho_h = Function(Vr)
    rho_averaged = Function(Vt)
    Vt_broken = FunctionSpace(state.mesh, BrokenElement(Vt.ufl_element()))
    rho_recoverer = Recoverer(rho0, rho_averaged, VDG=Vt_broken, boundary_method=method)
    w_h = Function(Vt)
    delta = 1.0

    # make expressions for determining water_v0
    pie = thermodynamics.pi(state.parameters, rho_averaged, theta0)
    p = thermodynamics.p(state.parameters, pie)
    T = thermodynamics.T(state.parameters, theta0, pie, water_v0)
    r_v_expr = thermodynamics.r_v(state.parameters, H, T, p)

    # make expressions to evaluate residual
    pi_ev = thermodynamics.pi(state.parameters, rho_averaged, theta0)
    p_ev = thermodynamics.p(state.parameters, pi_ev)
    T_ev = thermodynamics.T(state.parameters, theta0, pi_ev, water_v0)
    RH_ev = thermodynamics.RH(state.parameters, water_v0, T_ev, p_ev)
    RH = Function(Vt)

    for i in range(max_outer_solve_count):
        # solve for rho with theta_vd and w_v guesses
        compressible_hydrostatic_balance(state, theta0, rho_h, top=top,
                                         pi_boundary=pi_boundary, water_t=water_v0,
                                         solve_for_rho=True)

        # damp solution
        rho0.assign(rho0 * (1 - delta) + delta * rho_h)

        # calculate averaged rho
        rho_recoverer.project()

        RH.assign(RH_ev)
        if errornorm(RH, H) < 1e-10:
            break

        # now solve for r_v
        for j in range(max_inner_solve_count):
            w_h.interpolate(r_v_expr)
            water_v0.assign(water_v0 * (1 - delta) + delta * w_h)

            # compute theta_vd
            theta0.assign(theta_d * (1 + water_v0 / epsilon))

            # test quality of solution by re-evaluating expression
            RH.assign(RH_ev)
            if errornorm(RH, H) < 1e-10:
                break

        if i == max_outer_solve_count:
            raise RuntimeError('Hydrostatic balance solve has not converged within %i' % i, 'iterations')

    if pi0 is not None:
        pie = thermodynamics.pi(state.parameters, rho0, theta0)
        pi0.interpolate(pie)

    # do one extra solve for rho
    compressible_hydrostatic_balance(state, theta0, rho0, top=top,
                                     pi_boundary=pi_boundary,
                                     water_t=water_v0, solve_for_rho=True)
