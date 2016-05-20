from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function, TestFunction, TrialFunction, \
    LinearVariationalProblem, LinearVariationalSolver, FacetNormal, \
    dx, dot, grad, div, jump, avg, dS, dS_v, dS_h, action, inner, outer, sign, cross, CellNormal, lhs, rhs


class Advection(object):
    """
    Base class for advection schemes for dcore.

    :arg state: :class:`.State` object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        self.state = state
        self.ubar = Function(state.V[0])

    @abstractmethod
    def apply(self, x, x_out):
        """
        Function takes x as input, computes F(x) and returns x_out
        as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass


class NoAdvection(Advection):
    """
    An non-advection scheme that does nothing.
    """

    def apply(self, x_in, x_out):

        x_out.assign(x_in)


class LinearAdvection_Vt(Advection):
    """
    An advection scheme that uses the linearised background state
    in evaluation of the advection term for the Vt temperature space.

    :arg state: :class:`.State` object.
    :arg V:class:`.FunctionSpace` object. The Function space for temperature.
    :arg qbar: :class:`.Function` object. The reference function that we
    are linearising around.
    :arg options: a PETSc options dictionary
    """

    def __init__(self, state, V, qbar, options=None):
        super(LinearAdvection_Vt, self).__init__(state)

        p = TestFunction(V)
        q = TrialFunction(V)
        self.dq = Function(V)

        a = p*q*dx
        k = state.k             # Upward pointing unit vector
        L = -p*dot(self.ubar,k)*dot(k,grad(qbar))*dx

        aProblem = LinearVariationalProblem(a,L,self.dq)
        if options is None:
            options = {'ksp_type':'cg',
                       'pc_type':'bjacobi',
                       'sub_pc_type':'ilu'}

        self.solver = LinearVariationalSolver(aProblem,
                                              solver_parameters=options)

    def apply(self, x_in, x_out):
        dt = self.state.timestepping.dt
        self.solver.solve()
        x_out.assign(x_in + dt*self.dq)


class LinearAdvection_V3(Advection):
    """
    An advection scheme that uses the linearised background state
    in evaluation of the advection term for a DG space.

    :arg state: :class:`.State` object.
    :arg V:class:`.FunctionSpace` object. The DG Function space.
    :arg qbar: :class:`.Function` object. The reference function that we
    are linearising around.
    :arg options: a PETSc options dictionary
    """

    def __init__(self, state, V, qbar, options=None):
        super(LinearAdvection_V3, self).__init__(state)

        p = TestFunction(V)
        q = TrialFunction(V)
        self.dq = Function(V)

        n = FacetNormal(state.mesh)

        a = p*q*dx
        L = (dot(grad(p), self.ubar)*qbar*dx -
             jump(self.ubar*p, n)*avg(qbar)*(dS_v + dS_h))

        aProblem = LinearVariationalProblem(a,L,self.dq)
        if options is None:
            options = {'ksp_type':'cg',
                       'pc_type':'bjacobi',
                       'sub_pc_type':'ilu'}

        self.solver = LinearVariationalSolver(aProblem,
                                              solver_parameters=options)

    def apply(self, x_in, x_out):
        dt = self.state.timestepping.dt
        self.solver.solve()
        x_out.assign(x_in + dt*self.dq)


class DGAdvection(Advection):

    """
    DG 3 step SSPRK advection scheme that can be applied to a scalar
    or vector field

    :arg state: :class:`.State` object.
    :arg V: function space of advected field - should be DG
    :arg continuity: optional boolean.
         If ``True``, the advection equation is of the form:
         :math: `D_t +\nabla \cdot(uD) = 0`.
         If ``False``, the advection equation is of the form:
         :math: `D_t + (u \cdot \nabla)D = 0`.
    """

    def __init__(self, state, V, continuity=False):

        super(DGAdvection, self).__init__(state)

        element = V.fiat_element
        assert element.entity_dofs() == element.entity_closure_dofs(), "Provided space is not discontinuous"
        dt = state.timestepping.dt

        if V.extruded:
            surface_measure = (dS_h + dS_v)
        else:
            surface_measure = dS

        phi = TestFunction(V)
        D = TrialFunction(V)
        self.D1 = Function(V)
        self.dD = Function(V)

        n = FacetNormal(state.mesh)
        # ( dot(v, n) + |dot(v, n)| )/2.0
        un = 0.5*(dot(self.ubar, n) + abs(dot(self.ubar, n)))

        a_mass = inner(phi,D)*dx

        if continuity:
            a_int = -inner(grad(phi), outer(D, self.ubar))*dx
        else:
            a_int = -inner(div(outer(phi,self.ubar)),D)*dx

        a_flux = (dot(jump(phi), un('+')*D('+') - un('-')*D('-')))*surface_measure
        arhs = a_mass - dt*(a_int + a_flux)

        DGproblem = LinearVariationalProblem(a_mass, action(arhs,self.D1),
                                             self.dD)
        self.DGsolver = LinearVariationalSolver(DGproblem,
                                                solver_parameters={
                                                    'ksp_type':'preonly',
                                                    'pc_type':'bjacobi',
                                                    'sub_pc_type': 'ilu'})

    def apply(self, x_in, x_out):
        # SSPRK Stage 1
        self.D1.assign(x_in)
        self.DGsolver.solve()
        self.D1.assign(self.dD)

        # SSPRK Stage 2
        self.DGsolver.solve()
        self.D1.assign(0.75*x_in + 0.25*self.dD)

        # SSPRK Stage 3
        self.DGsolver.solve()

        x_out.assign((1.0/3.0)*x_in + (2.0/3.0)*self.dD)


class EmbeddedDGAdvection(DGAdvection):

    def __init__(self, state, Vdg, continuity):

        super(EmbeddedDGAdvection, self).__init__(state, Vdg, continuity)

        self.xdg_in = Function(Vdg)
        self.xdg_out = Function(Vdg)

    def apply(self, x_in, x_out):

        self.xdg_in.interpolate(x_in)
        super(EmbeddedDGAdvection, self).apply(self.xdg_in, self.xdg_out)
        x_out.project(self.xdg_out)


class EulerPoincareForm(Advection):

    def __init__(self, state, V):
        super(EulerPoincareForm, self).__init__(state)

        dt = state.timestepping.dt
        w = TestFunction(V)
        u = TrialFunction(V)
        self.u0 = Function(V)
        ustar = 0.5*(self.u0 + u)
        n = FacetNormal(state.mesh)
        Upwind = 0.5*(sign(dot(self.ubar, n))+1)

        outward_normals = CellNormal(state.mesh)
        perp = lambda u: cross(outward_normals, u)
        perp_u_upwind = Upwind('+')*cross(outward_normals('+'),ustar('+')) + Upwind('-')*cross(outward_normals('-'),ustar('-'))
        Eqn = (
            (inner(w, u-self.u0)
             - dt*inner(w, div(perp(ustar))*perp(self.ubar))
             - dt*div(w)*inner(ustar, self.ubar))*dx
            - dt*inner(jump(inner(w, perp(self.ubar)),n), perp_u_upwind)*dS
            + dt*jump(inner(w, perp(self.ubar))*perp(ustar), n)*dS
        )
        a = lhs(Eqn)
        L = rhs(Eqn)
        self.u1 = Function(V)
        uproblem = LinearVariationalProblem(a, L, self.u1)
        self.usolver = LinearVariationalSolver(uproblem)

    def apply(self, x_in, x_out):
        self.u0.assign(x_in)
        self.usolver.solve()
        x_out.assign(self.u1)
