"""
A module containing some tools for computing initial conditions, such
as balanced initial conditions.
"""

from __future__ import absolute_import
from firedrake import MixedFunctionSpace, TrialFunctions, TestFunctions, \
    FacetNormal, inner, div, dx, ds_b, ds_t, DirichletBC, \
    Expression, Function, \
    LinearVariationalProblem, LinearVariationalSolver


def compressible_hydrostatic_balance(state, theta0, rho0,
                                     top=True, rho_boundary=0,
                                     params=None):
    """
    Compute a hydrostatically balanced density given a potential temperature
    profile.

    :arg state: The :class:`State` object.
    :arg theta0: :class:`.Function`containing the potential temperature.
    :arg rho0: :class:`.Function` to write the initial density into.
    :arg top: If True, set a boundary condition at the top. Otherwise, set
    it at the bottom.
    :arg rho_boundary: a field or expression to use as boundary data on
    the top or bottom as specified.
    """

    # Calculate hydrostatic Pi
    W = MixedFunctionSpace((state.Vv,state.V[1]))
    v, pi = TrialFunctions(W)
    dv, dpi = TestFunctions(W)

    n = FacetNormal(state.mesh)

    cp = state.parameters.cp
    g = state.parameters.g

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

    k = state.k
    arhs = (
        - g*inner(dv,k)*dx
        - cp*inner(dv,n)*theta0*rho_boundary*bmeasure
    )

    bcs = [DirichletBC(W.sub(0), Expression(("0.", "0.")), bstring)]

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
                  'fieldsplit_0_pc_type': 'bjacobi',
                  'fieldsplit_0_sub_pc_type': 'ilu',
                  'fieldsplit_1_ksp_type': 'richardson',
                  'fieldsplit_1_ksp_max_it': 5,
                  "fieldsplit_1_ksp_monitor_true_residual": True,
                  'fieldsplit_1_pc_type': 'bjacobi',
                  'fieldsplit_1_sub_pc_type': 'ilu'}

    PiSolver = LinearVariationalSolver(PiProblem,
                                       solver_parameters=params)

    PiSolver.solve()
    v, Pi = w.split()

    kappa = state.kappa
    R_d = state.R_d
    p_0 = state.p_0

    rho0.interpolate(p_0*(Pi**((1-kappa)/kappa))/R_d/theta0)
