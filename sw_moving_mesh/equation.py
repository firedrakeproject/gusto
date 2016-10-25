from firedrake import Function, TestFunction, TrialFunction, FacetNormal, inner, outer, div, grad, perp, dx, dot, jump, dS, CellNormal, cross, sign

class Equation(object):

    def __init__(self, state, ubar):
        self.state = state
        self.ubar = ubar

class Advection(Equation):

    def __init__(self, state, ubar, V):
        super(Advection, self).__init__(state, ubar)
        self.test = TestFunction(V)
        self.trial = TrialFunction(V)
        self.q = Function(V)
        self.n = FacetNormal(V.mesh())

    def mass_term(self, q, dx):
        return inner(self.test, q)*dx

    def advection_term(self, q, dx, dS):
        un = 0.5*(dot(self.ubar, self.n) + abs(dot(self.ubar, self.n)))
        return -inner(grad(self.test), outer(q,self.ubar))*dx + dot(jump(self.test), un('+')*q('+') - un('-')*q('-'))*dS

class EulerPoincareMomentum(Equation):

    def __init__(self, state, ubar, V):
        super(EulerPoincareMomentum, self).__init__(state, ubar)
        self.test = TestFunction(V)
        self.trial = TrialFunction(V)
        self.q = Function(V)
        self.n = FacetNormal(V.mesh())
        self.outward_normals = CellNormal(V.mesh())

    def mass_term(self, q, dx):
        return inner(self.test, q)*dx

    def advection_term(self, q, dx, dS):
        Upwind = 0.5*(sign(dot(self.ubar, self.n))+1)
        outward_normals = self.outward_normals
        perp = lambda u: cross(outward_normals, u)
        perp_u_upwind = Upwind('+')*cross(outward_normals('+'),q('+')) + Upwind('-')*cross(outward_normals('-'),q('-'))
        return (-inner(self.test, div(perp(q))*perp(self.ubar)) - div(self.test)*inner(q, self.ubar))*dx - inner(jump(inner(self.test, perp(self.ubar)), self.n), perp_u_upwind)*dS + jump(inner(self.test, perp(self.ubar))*perp(q), self.n)*dS


