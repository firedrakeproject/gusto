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
    sin, cos, sqrt, asin, atan_2, as_vector, Min, Max, exp, warning
from gusto.forcing import exner


__all__ = ["latlon_coords", "sphere_to_cartesian", "incompressible_hydrostatic_balance", "compressible_hydrostatic_balance", "remove_initial_w", "eady_initial_v", "compressible_eady_initial_v", "calculate_Pi0", "moist_hydrostatic_balance"]


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

    bcs = [DirichletBC(WV[0], 0.0, bstring)]

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
    profile.

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
    g = state.parameters.g
    arhs -= g*inner(dv, state.k)*dx

    bcs = [DirichletBC(W.sub(0), 0.0, bstring)]

    w = Function(W)
    PiProblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)

    if params is None:
        params = {'pc_type': 'fieldsplit',
                  'pc_fieldsplit_type': 'schur',
                  'ksp_type': 'gmres',
                  'ksp_monitor_true_residual': True,
                  'ksp_max_it': 100,
                  'ksp_gmres_restart': 50,
                  'pc_fieldsplit_schur_fact_type': 'FULL',
                  'pc_fieldsplit_schur_precondition': 'selfp',
                  'fieldsplit_0_ksp_type': 'richardson',
                  'fieldsplit_0_ksp_max_it': 5,
                  'fieldsplit_0_pc_type': 'gamg',
                  'fieldsplit_1_pc_gamg_sym_graph': True,
                  'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                  'fieldsplit_1_mg_levels_ksp_chebyshev_esteig': True,
                  'fieldsplit_1_mg_levels_ksp_max_it': 5,
                  'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                  'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

    PiSolver = LinearVariationalSolver(PiProblem,
                                       solver_parameters=params)

    PiSolver.solve()
    v, Pi = w.split()
    if pi0 is not None:
        pi0.assign(Pi)

    kappa = state.parameters.kappa
    R_d = state.parameters.R_d
    p_0 = state.parameters.p_0

    if solve_for_rho:
        w1 = Function(W)
        v, rho = w1.split()
        rho.interpolate(p_0*(Pi**((1-kappa)/kappa))/R_d/theta0)
        v, rho = split(w1)
        dv, dpi = TestFunctions(W)
        pi = ((R_d/p_0)*rho*theta0)**(kappa/(1.-kappa))
        F = (
            (cp*inner(v, dv) - cp*div(dv*theta)*pi)*dx
            + dpi*div(theta0*v)*dx
            + cp*inner(dv, n)*theta*pi_boundary*bmeasure
        )
        F += g*inner(dv, state.k)*dx
        rhoproblem = NonlinearVariationalProblem(F, w1, bcs=bcs)
        rhosolver = NonlinearVariationalSolver(rhoproblem, solver_parameters=params)
        rhosolver.solve()
        v, rho_ = w1.split()
        rho0.assign(rho_)
    else:
        rho0.interpolate(p_0*(Pi**((1-kappa)/kappa))/R_d/theta0)


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
    Pi_exp = exner(theta0, rho0, state)
    Pi = Function(Vr).interpolate(Pi_exp)

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
    Pi_exp = exner(theta0, rho0, state)
    Pi = Function(Vr).interpolate(Pi_exp)
    Pi0 = assemble(Pi*dx)/assemble(Constant(1)*dx(domain=state.mesh))

    return Pi0


def moist_hydrostatic_balance(state, theta_e, water_t, pi_boundary=Constant(1.0)):
    """
    Given a wet equivalent potential temperature, theta_e, and the total moisture
    content, water_t, compute a hydrostatically balance virtual potential temperature,
    dry density and water vapour profile.
    :arg state: The :class:`State` object.
    :arg theta_e: The initial wet equivalent potential temperature profile.
    :arg water_t: The total water pseudo-mixing ratio profile.
    :arg pi_boundary: the value of pi on the lower boundary of the domain.
    """

    theta0 = state.fields('theta')
    rho0 = state.fields('rho')
    water_v0 = state.fields('water_v')

    # Calculate hydrostatic Pi
    Vt = theta0.function_space()
    Vr = rho0.function_space()
    Vv = state.spaces("Vv")
    n = FacetNormal(state.mesh)

    param = state.parameters

    # define some parameters as attributes
    R_d = param.R_d
    p_0 = param.p_0
    cp = param.cp
    c_pv = param.c_pv
    c_pl = param.c_pl
    R_v = param.R_v
    L_v0 = param.L_v0
    T_0 = param.T_0
    w_sat1 = param.w_sat1
    w_sat2 = param.w_sat2
    w_sat3 = param.w_sat3
    w_sat4 = param.w_sat4
    g = param.g

    VDG = state.spaces("DG")
    if any(deg > 2 for deg in VDG.ufl_element().degree()):
        warning("default quadrature degree most likely not sufficient for this degree element")
    quadrature_degree = (5, 5)

    params = {'ksp_type': 'preonly',
              'ksp_monitor_true_residual': True,
              'ksp_converged_reason': True,
              'snes_converged_reason': True,
              'ksp_max_it': 100,
              'mat_type': 'aij',
              'pc_type': 'lu',
              'pc_factor_mat_solver_package': 'mumps'}

    theta0.interpolate(theta_e)
    water_v0.interpolate(water_t)
    Pi = Function(Vr)
    epsilon = 0.9  # relaxation constant

    # set up mixed space
    Z = MixedFunctionSpace((Vt, Vt))
    z = Function(Z)

    gamma, phi = TestFunctions(Z)

    theta_v, w_v = z.split()

    # give first guesses for trial functions
    theta_v.assign(theta0)
    w_v.assign(water_v0)

    theta_v, w_v = split(z)

    # define variables
    T = Pi * theta_v / (1 + w_v * R_v / R_d)
    p = p_0 * Pi ** (cp / R_d)
    L_v = L_v0 - (c_pl - c_pv) * (T - T_0)
    w_sat = w_sat1 / (p * exp(w_sat2 * (T - T_0) / (T - w_sat3)) - w_sat4)

    dxp = dx(degree=(quadrature_degree))

    # set up weak form of theta_e and w_sat equations
    F = (-gamma * theta_e * dxp
         + gamma * T * (p / (p_0 * (1 + w_v * R_v / R_d))) **
         (- R_d / (cp + c_pl * water_t)) *
         exp(L_v * w_v / ((cp + c_pl * water_t) * T)) * dxp
         - phi * w_v * dxp
         + phi * w_sat * dxp)

    problem = NonlinearVariationalProblem(F, z)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)

    theta_v, w_v = z.split()

    Pi_h = Function(Vr).interpolate((p / p_0) ** (R_d / cp))

    # solve for Pi with theta_v and w_v constant
    # then solve for theta_v and w_v with Pi constant
    for i in range(5):
        compressible_hydrostatic_balance(state, theta0, rho0, pi0=Pi_h, water_t=water_t)
        Pi.assign(Pi * (1 - epsilon) + epsilon * Pi_h)
        solver.solve()
        theta0.assign(theta0 * (1 - epsilon) + epsilon * theta_v)
        water_v0.assign(water_v0 * (1 - epsilon) + epsilon * w_v)

    # now begin on Newton solver, setup up new mixed space
    Z = MixedFunctionSpace((Vt, Vt, Vr, Vv))
    z = Function(Z)

    gamma, phi, psi, w = TestFunctions(Z)

    theta_v, w_v, pi, v = z.split()

    # use previous values as first guesses for newton solver
    theta_v.assign(theta0)
    w_v.assign(water_v0)
    pi.assign(Pi)

    theta_v, w_v, pi, v = split(z)

    # define variables
    T = pi * theta_v / (1 + w_v * R_v / R_d)
    p = p_0 * pi ** (cp / R_d)
    L_v = L_v0 - (c_pl - c_pv) * (T - T_0)
    w_sat = w_sat1 / (p * exp(w_sat2 * (T - T_0) / (T - w_sat3)) - w_sat4)

    F = (-gamma * theta_e * dxp
         + gamma * T * (p / (p_0 * (1 + w_v * R_v / R_d))) **
         (- R_d / (cp + c_pl * water_t)) *
         exp(L_v * w_v / ((cp + c_pl * water_t) * T)) * dxp
         - phi * w_v * dxp
         + phi * w_sat * dxp
         + cp * inner(v, w) * dxp
         - cp * div(w * theta_v / (1.0 + water_t)) * pi * dxp
         + psi * div(theta_v * v / (1.0 + water_t)) * dxp
         + cp * inner(w, n) * pi_boundary * theta_v / (1.0 + water_t) * ds_b
         + g * inner(w, state.k) * dxp)

    bcs = [DirichletBC(Z.sub(3), 0.0, "top")]

    problem = NonlinearVariationalProblem(F, z, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)

    solver.solve()

    theta_v, w_v, pi, v = z.split()

    # assign final values
    theta0.assign(theta_v)
    water_v0.assign(w_v)

    # find rho
    compressible_hydrostatic_balance(state, theta0, rho0, water_t=water_t, solve_for_rho=True)
