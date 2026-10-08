from firedrake import TestFunction, TrialFunction, Function, \
    dot, grad, dx, VectorSpaceBasis, solve, TestFunctions, TrialFunctions, \
    inner, div, Constant, assemble


def nondivergent_flow(equation, zeta0, u0, D0):
    """
    Returns u0 and D0, balanced velocity and depth fields, given a
    vorticity field zeta0. Balance is defined as

    Args:
        equation (:class:`PrognosticEquation`): the model's equation object.
        zeta0 (:class:`ufl.Expr`): the input vorticity field.
        u0 (:class:`Function`): the velocity to be returned.
        D0 (:class:`Function`): the depth to be returned.
    """

    domain = equation.domain
    Vcg = domain.spaces("H1")

    # compute initial streamfunction from vorticity by solving Poisson equation
    v = TestFunction(Vcg)
    p = TrialFunction(Vcg)
    psi = Function(Vcg)
    a = -dot(grad(v), grad(p)) * dx
    L = v * zeta0 * dx
    nullspace = VectorSpaceBasis(constant=True)
    solve(a == L, psi, nullspace=nullspace,
          solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

    # compute initial velocity from streamfunction
    u0.project(domain.perp(grad(psi)))

    # solve mixed Poisson problem for (v, depth) with v=u_t and
    # div(v)=0 so that we don't generate any divergence initially
    VHdiv = domain.spaces("HDiv")
    Vdg = domain.spaces("L2")
    W = VHdiv * Vdg
    v, h = TrialFunctions(W)
    p, q = TestFunctions(W)
    g = equation.parameters.g
    f = equation.prescribed_fields("coriolis")
    a = inner(p, v) * dx - g * div(p) * h * dx + q * div(v) * dx
    L = (
        -(f + zeta0) * inner(p, domain.perp(u0)) * dx
        + 0.5 * div(p) * dot(u0, u0) * dx
    )
    w = Function(W)
    solve(a == L, w, nullspace=nullspace)
    _, D = w.subfunctions
    D0.assign(D)

    # adjust depth to have initial mean of H as set in the parameters
    C = Function(Vdg).assign(Constant(1.0))
    area = assemble(C*dx)
    Dmean = assemble(D*dx)/area
    D0.assign(D0 - Dmean + equation.parameters.H)
