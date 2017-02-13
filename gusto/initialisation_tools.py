"""
A module containing some tools for computing initial conditions, such
as balanced initial conditions.
"""

from __future__ import absolute_import
from firedrake import MixedFunctionSpace, TrialFunctions, TestFunctions, \
    FacetNormal, inner, div, dx, ds_b, ds_t, DirichletBC, \
    Expression, Function, Constant, \
    LinearVariationalProblem, LinearVariationalSolver, \
    NonlinearVariationalProblem, NonlinearVariationalSolver, split


def compressible_hydrostatic_balance(state, theta0, rho0, pi0=None,
                                     top=False, pi_boundary=Constant(1.0),
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
    """

    # Calculate hydrostatic Pi
    V1 = state.spaces("DG")
    W = MixedFunctionSpace((state.Vv, V1))
    v, pi = TrialFunctions(W)
    dv, dpi = TestFunctions(W)

    n = FacetNormal(state.mesh)

    cp = state.parameters.cp

    alhs = (
        (cp*inner(v,dv) - cp*div(dv*theta0)*pi)*dx
        + dpi*div(theta0*v)*dx
    )

    if top:
        bmeasure = ds_t
        bstring = "bottom"
    else:
        bmeasure = ds_b
        bstring = "top"

    arhs = -cp*inner(dv,n)*theta0*pi_boundary*bmeasure
    if state.geopotential_form:
        Phi = state.Phi
        arhs += div(dv)*Phi*dx - inner(dv,n)*Phi*bmeasure
    else:
        g = state.parameters.g
        arhs -= g*inner(dv,state.k)*dx

    if(state.mesh.geometric_dimension() == 2):
        bcs = [DirichletBC(W.sub(0), Expression(("0.", "0.")), bstring)]
    elif(state.mesh.geometric_dimension() == 3):
        bcs = [DirichletBC(W.sub(0), Expression(("0.", "0.", "0.")), bstring)]
    w = Function(W)
    PiProblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)

    if(params is None):
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
                  'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                  'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
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
            (cp*inner(v,dv) - cp*div(dv*theta0)*pi)*dx
            + dpi*div(theta0*v)*dx
            + cp*inner(dv,n)*theta0*pi_boundary*bmeasure
        )
        if state.geopotential_form:
            F += - div(dv)*Phi*dx + inner(dv,n)*Phi*bmeasure
        else:
            F += g*inner(dv,state.k)*dx
        rhoproblem = NonlinearVariationalProblem(F, w1, bcs=bcs)
        rhosolver = NonlinearVariationalSolver(rhoproblem, solver_parameters=params)
        rhosolver.solve()
        v, rho_ = w1.split()
        rho0.assign(rho_)
    else:
        rho0.interpolate(p_0*(Pi**((1-kappa)/kappa))/R_d/theta0)


def remove_initial_w(u, Vv):
    bc = DirichletBC(u.function_space()[0], Constant((0,0)), "bottom")
    bc.apply(u)
    uv = Function(Vv).project(u)
    ustar = Function(u.function_space()).project(uv)
    uin = Function(u.function_space()).assign(u - ustar)
    u.assign(uin)
