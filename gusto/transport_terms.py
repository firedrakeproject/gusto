from firedrake import dx, dot, grad, div, jump, avg, inner, \
    outer, sign, cross, CellNormal, curl, dS
from gusto.terms import Term


__all__ = ["LinearAdvectionTerm", "AdvectionTerm", "VectorInvariantTerm"]


class TransportTerm(Term):
    """
    Base class for transport terms.

    The equation is assumed to be in the form:

    q_t + L(q) = 0

    where q is the (scalar or vector) field to be solved for.

    :arg state: :class:`.State` object.
    :arg test: :class:`.TestFunction` object.
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    """

    def __init__(self, state, test, *, ibp="once"):

        super().__init__(state, test)

        self.ibp = ibp

        # **** until other options are fixed ****
        self.dS = dS


class LinearAdvectionTerm(TransportTerm):
    """
    Class for linear transport equation.

    :arg state: :class:`.State` object.
    :arg test: :class:`.TestFunction` object.
    :arg qbar: The reference function that the equation has been linearised
               about. It is assumed that the reference velocity is zero and
               the ubar below is the nonlinear advecting velocity
               0.5*(u'^(n+1) + u'(n)))
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    :arg equation_form: (optional) string, can take the values 'continuity',
                        which means the equation is in continuity form
                        L(q) = div(u'*qbar), or 'advective', which means the
                        equation is in advective form L(q) = u' dot grad(qbar).
                        Default is "advective"
    """

    def __init__(self, state, test, qbar, ibp=None, equation_form="advective"):
        super().__init__(state=state, test=test, ibp=ibp)
        if equation_form == "advective" or equation_form == "continuity":
            self.continuity = (equation_form == "continuity")
        else:
            raise ValueError("equation_form must be either 'advective' or 'continuity', not %s" % equation_form)

        self.qbar = qbar

        # currently only used with the following option combinations:
        if self.continuity and ibp is not "once":
            raise NotImplementedError("If we are solving a linear continuity equation, we integrate by parts once")
        if not self.continuity and ibp is not None:
            raise NotImplementedError("If we are solving a linear advection equation, we do not integrate by parts.")

    def evaluate(self, q, fields):

        uadv = self.state.fields("uadv")
        if self.continuity:
            L = (-dot(grad(self.test), uadv)*self.qbar*dx +
                 jump(uadv*self.test, self.n)*avg(self.qbar)*self.dS)
        else:
            L = self.test*dot(uadv, self.state.k)*dot(self.state.k, grad(self.qbar))*dx
        return L


class AdvectionTerm(TransportTerm):
    """
    Class for discretisation of the transport equation.

    :arg state: :class:`.State` object.
    :arg test: :class:`.TestFunction` object.
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    :arg equation_form: (optional) string, can take the values 'continuity',
                        which means the equation is in continuity form
                        L(q) = div(u*q), or 'advective', which means the
                        equation is in advective form L(q) = u dot grad(q).
                        Default is "advective"
    :arg vector_manifold: Boolean. If true adds extra terms that are needed for
    advecting vector equations on manifolds.
    """
    def __init__(self, state, test, *, ibp="once", equation_form="advective",
                 vector_manifold=False, outflow=False):
        super().__init__(state=state, test=test, ibp=ibp)
        if equation_form == "advective" or equation_form == "continuity":
            self.continuity = (equation_form == "continuity")
        else:
            raise ValueError("equation_form must be either 'advective' or 'continuity'")
        self.vector_manifold = vector_manifold
        self.outflow = outflow
        if outflow and ibp is None:
            raise ValueError("outflow is True and ibp is None are incompatible options")

    def evaluate(self, q, fields):

        uadv = self.state.fields("uadv")
        un = 0.5*(dot(uadv, self.n) + abs(dot(uadv, self.n)))

        if self.continuity:
            if self.ibp == "once":
                L = -inner(grad(self.test), outer(q, uadv))*dx
            else:
                L = inner(self.test, div(outer(q, uadv)))*dx
        else:
            if self.ibp == "once":
                L = -inner(div(outer(self.test, uadv)), q)*dx
            else:
                L = inner(outer(self.test, uadv), grad(q))*dx

        if self.dS is not None and self.ibp is not None:
            L += dot(jump(self.test), (un('+')*q('+')
                                       - un('-')*q('-')))*self.dS
            if self.ibp == "twice":
                L -= (inner(self.test('+'),
                            dot(uadv('+'), self.n('+'))*q('+'))
                      + inner(self.test('-'),
                              dot(uadv('-'), self.n('-'))*q('-')))*self.dS

        if self.outflow:
            L += self.test*self.un*q*self.ds

        if self.vector_manifold:
            w = self.test
            u = q
            n = self.n
            dS = self.dS
            L += un('+')*inner(w('-'), n('+')+n('-'))*inner(u('+'), n('+'))*dS
            L += un('-')*inner(w('+'), n('+')+n('-'))*inner(u('-'), n('-'))*dS
        return L


class VectorInvariantTerm(TransportTerm):
    """
    Class defining the vector invariant form of the vector advection equation.

    :arg state: :class:`.State` object.
    :arg test: :class:`.TestFunction` object.
    :arg ibp: (optional) string, stands for 'integrate by parts' and can
              take the value None, "once" or "twice". Defaults to "once".
    """
    def __init__(self, state, test, *, ibp="once"):
        super().__init__(state=state, test=test, ibp=ibp)

        if self.state.mesh.topological_dimension() == 3 and ibp == "twice":
                raise NotImplementedError("ibp=twice is not implemented for 3d problems")

    def evaluate(self, q, fields):

        uadv = self.state.fields("uadv")
        Upwind = 0.5*(sign(dot(uadv, self.n))+1)

        if self.state.mesh.topological_dimension() == 3:
            # <w,curl(u) cross ubar + grad( u.ubar)>
            # =<curl(u),ubar cross w> - <div(w), u.ubar>
            # =<u,curl(ubar cross w)> -
            #      <<u_upwind, [[n cross(ubar cross w)cross]]>>

            both = lambda u: 2*avg(u)

            L = (
                inner(q, curl(cross(uadv, self.test)))*dx
                - inner(both(Upwind*q),
                        both(cross(self.n, cross(uadv, self.test))))*self.dS
            )

        else:
            perp = self.state.perp
            if self.state.on_sphere:
                outward_normals = CellNormal(self.state.mesh)
                perp_u_upwind = lambda q: Upwind('+')*cross(outward_normals('+'), q('+')) + Upwind('-')*cross(outward_normals('-'), q('-'))
            else:
                perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))
            if self.ibp == "once":
                gradperp = lambda u: perp(grad(u))
                L = (
                    -inner(gradperp(inner(self.test, perp(uadv))), q)*dx
                    - inner(jump(inner(self.test, perp(uadv)), self.n),
                            perp_u_upwind(q))*self.dS
                )
            else:
                L = (
                    (-inner(self.test, div(perp(q))*perp(uadv)))*dx
                    - inner(jump(inner(self.test, perp(uadv)), self.n),
                            perp_u_upwind(q))*self.dS
                    + jump(inner(self.test,
                                 perp(uadv))*perp(q), self.n)*self.dS
                )

        L -= 0.5*div(self.test)*inner(q, uadv)*dx

        return L
