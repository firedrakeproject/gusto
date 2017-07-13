from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function, TestFunction, TrialFunction, \
    FacetNormal, \
    dx, dot, grad, div, jump, avg, dS, dS_v, dS_h, inner, \
    outer, sign, cross, CellNormal, sqrt, Constant, \
    curl, BrokenElement, FunctionSpace


class TransportEquation(object):
    """
    Base class for transport equations in Gusto.

    The equation is assumed to be in the form:

    q_t + L(q) = 0

    where q is the (scalar or vector) field to be solved for.

    :arg state: :class:`.State` object.
    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, V, ibp="once", solver_params=None):
        self.state = state
        self.V = V
        self.ibp = ibp

        # set up functions required for forms
        self.ubar = Function(state.spaces("HDiv"))
        self.test = TestFunction(V)
        self.trial = TrialFunction(V)

        # find out if we are CG
        nvertex = V.ufl_domain().ufl_cell().num_vertices()
        entity_dofs = V.finat_element.entity_dofs()
        # If there are as many dofs on vertices as there are vertices,
        # assume a continuous space.
        try:
            self.is_cg = sum(map(len, entity_dofs[0].values())) == nvertex
        except KeyError:
            self.is_cg = sum(map(len, entity_dofs[(0, 0)].values())) == nvertex

        # DG, embedded DG and hybrid SUPG methods need surface measures,
        # n and un
        if self.is_cg:
            self.dS = None
        else:
            if V.extruded:
                self.dS = (dS_h + dS_v)
            else:
                self.dS = dS
            self.n = FacetNormal(state.mesh)
            self.un = 0.5*(dot(self.ubar, self.n) + abs(dot(self.ubar, self.n)))

        if solver_params:
            self.solver_parameters = solver_params

        # default solver options
        else:
            self.solver_parameters = {'ksp_type': 'cg',
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}

    def mass_term(self, q):
        return inner(self.test, q)*dx

    @abstractmethod
    def advection_term(self):
        pass


class LinearAdvection(TransportEquation):
    """
    Class for linear transport equation.

    :arg state: :class:`.State` object.
    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
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
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """

    def __init__(self, state, V, qbar, ibp=None, equation_form="advective", solver_params=None):
        super(LinearAdvection, self).__init__(state, V, ibp, solver_params)
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

        # default solver options
        self.solver_parameters = {'ksp_type': 'cg',
                                  'pc_type': 'bjacobi',
                                  'sub_pc_type': 'ilu'}

    def advection_term(self, q):

        if self.continuity:
            L = (-dot(grad(self.test), self.ubar)*self.qbar*dx +
                 jump(self.ubar*self.test, self.n)*avg(self.qbar)*self.dS)
        else:
            L = self.test*dot(self.ubar, self.state.k)*dot(self.state.k, grad(self.qbar))*dx
        return L


class AdvectionEquation(TransportEquation):
    """
    Class for discretisation of the transport equation.

    :arg state: :class:`.State` object.
    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    :arg equation_form: (optional) string, can take the values 'continuity',
                        which means the equation is in continuity form
                        L(q) = div(u*q), or 'advective', which means the
                        equation is in advective form L(q) = u dot grad(q).
                        Default is "advective"
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """
    def __init__(self, state, V, ibp="once", equation_form="advective", solver_params=None):
        super(AdvectionEquation, self).__init__(state, V, ibp, solver_params)
        if equation_form == "advective" or equation_form == "continuity":
            self.continuity = (equation_form == "continuity")
        else:
            raise ValueError("equation_form must be either 'advective' or 'continuity'")

    def advection_term(self, q):

        if self.continuity:
            if self.ibp == "once":
                L = -inner(grad(self.test), outer(q, self.ubar))*dx
            else:
                L = inner(self.test, div(outer(q, self.ubar)))*dx
        else:
            if self.ibp == "once":
                L = -inner(div(outer(self.test, self.ubar)), q)*dx
            else:
                L = inner(outer(self.test, self.ubar), grad(q))*dx

        if self.dS is not None and self.ibp is not None:
            L += dot(jump(self.test), (self.un('+')*q('+')
                                       - self.un('-')*q('-')))*self.dS
            if self.ibp == "twice":
                L -= (inner(self.test('+'), dot(self.ubar('+'), self.n('+'))*q('+'))
                      + inner(self.test('-'), dot(self.ubar('-'),
                                                  self.n('-'))*q('-')))*self.dS

        return L


class EmbeddedDGAdvection(AdvectionEquation):
    """
    Class for the transport equation, using an embedded DG advection scheme.

    :arg state: :class:`.State` object.
    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg ibp: (optional) string, stands for 'integrate by parts' and can take
              the value None, "once" or "twice". Defaults to "once".
    :arg equation_form: (optional) string, can take the values 'continuity',
                        which means the equation is in continuity form
                        L(q) = div(u*q), or 'advective', which means the
                        equation is in advective form L(q) = u dot grad(q).
                        Default is "advective"
    :arg Vdg: (optional) :class:`.FunctionSpace object. The embedding function
              space. Defaults to None which means that a broken space is
              constructed for you.
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """

    def __init__(self, state, V, ibp="once", equation_form="advective", Vdg=None, solver_params=None):

        if Vdg is None:
            # Create broken space, functions and projector
            V_elt = BrokenElement(V.ufl_element())
            self.space = FunctionSpace(state.mesh, V_elt)
        else:
            self.space = Vdg

        super(EmbeddedDGAdvection, self).__init__(state, self.space, ibp, equation_form, solver_params)


class SUPGAdvection(AdvectionEquation):
    """
    Class for the transport equation.

    :arg state: :class:`.State` object.
    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg ibp: (optional) string, stands for 'integrate by parts' and can
              take the value None, "once" or "twice". Defaults to "twice"
              since we commonly use this scheme for parially continuous
              spaces, in which case we don't want to take a derivative of
              the test function. If using for a fully continuous space, we
              don't integrate by parts at all (so you can set ibp=None).
    :arg equation_form: (optional) string, can take the values 'continuity',
                        which means the equation is in continuity form
                        L(q) = div(u*q), or 'advective', which means the
                        equation is in advective form L(q) = u dot grad(q).
                        Default is "advective"
    :arg supg_params: (optional) dictionary of parameters for the SUPG method.
                      Can contain:
                      'ax', 'ay', 'az', which specify the coefficients in
                      the x, y, z directions respectively
                      'dg_direction', which can be 'horizontal' or 'vertical',
                      and specifies the direction in which the function space
                      is discontinuous so that we can apply DG upwinding in
                      this direction.
                      Appropriate defaults are provided for these parameters,
                      in particular, the space is assumed to be continuous.
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """
    def __init__(self, state, V, ibp="twice", equation_form="advective", supg_params=None, solver_params=None):

        if not solver_params:
            # SUPG method leads to asymmetric matrix (since the test function
            # is effectively modified), so don't use CG
            solver_params = {'ksp_type': 'gmres',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}

        super(SUPGAdvection, self).__init__(state, V, ibp, equation_form, solver_params)

        # if using SUPG we either integrate by parts twice, or not at all
        if ibp == "once":
            raise ValueError("if using SUPG we don't integrate by parts once")
        if ibp is None and not self.is_cg:
            raise ValueError("are you very sure you don't need surface terms?")

        # set default SUPG parameters
        dt = state.timestepping.dt
        supg_params = supg_params.copy() if supg_params else {}
        supg_params.setdefault('ax', dt/sqrt(15.))
        supg_params.setdefault('ay', dt/sqrt(15.))
        supg_params.setdefault('az', dt/sqrt(15.))
        # default assumes a continuous space
        supg_params.setdefault('dg_direction', None)

        # find out if we need to do DG upwinding in any direction and set
        # self.dS accordingly
        if supg_params["dg_direction"] is None:
            # space is assumed to be continuous and we don't need
            # any interior surface integrals
            self.dS = None
        elif supg_params["dg_direction"] == "horizontal":
            # if space is discontinuous in the horizontal direction, we
            # need to include surface integrals on the vertical faces
            self.dS = dS_v
        elif supg_params["dg_direction"] == "vertical":
            # if space is discontinuous in the vertical direction, we
            # need to include surface integrals on the horizontal faces
            self.dS = dS_h
        else:
            raise RuntimeError("Invalid dg_direction in supg_params.")

        # make SUPG test function
        if state.mesh.topological_dimension() == 2:
            taus = [supg_params["ax"], supg_params["ay"]]
            if supg_params["dg_direction"] == "horizontal":
                taus[0] = 0.0
            elif supg_params["dg_direction"] == "vertical":
                taus[1] = 0.0
            tau = Constant(((taus[0], 0.), (0., taus[1])))
        elif state.mesh.topological_dimension() == 3:
            taus = [supg_params["ax"], supg_params["ay"], supg_params["az"]]
            if supg_params["dg_direction"] == "horizontal":
                taus[0] = 0.0
                taus[1] = 0.0
            elif supg_params["dg_direction"] == "vertical":
                taus[2] = 0.0

            tau = Constant(((taus[0], 0., 0.), (0., taus[1], 0.), (0., 0., taus[2])))
        dtest = dot(dot(self.ubar, tau), grad(self.test))
        self.test += dtest


class VectorInvariant(TransportEquation):
    """
    Class defining the vector invariant form of the vector advection equation.

    :arg state: :class:`.State` object.
    :arg V: Function space
    :arg ibp: (optional) string, stands for 'integrate by parts' and can
              take the value None, "once" or "twice". Defaults to "once".
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """
    def __init__(self, state, V, ibp="once", solver_params=None):
        super(VectorInvariant, self).__init__(state, V, ibp, solver_params)

        self.Upwind = 0.5*(sign(dot(self.ubar, self.n))+1)

        if self.state.mesh.topological_dimension() == 2:
            self.perp = state.perp
            if V.extruded:
                self.perp_u_upwind = lambda q: self.Upwind('+')*state.perp(q('+')) + self.Upwind('-')*state.perp(q('-'))
            else:
                outward_normals = CellNormal(state.mesh)
                self.perp_u_upwind = lambda q: self.Upwind('+')*cross(outward_normals('+'), q('+')) + self.Upwind('-')*cross(outward_normals('-'), q('-'))
            self.gradperp = lambda u: state.perp(grad(u))
        elif self.state.mesh.topological_dimension() == 3:
            if self.ibp == "twice":
                raise NotImplementedError("ibp=twice is not implemented for 3d problems")
        else:
            raise RuntimeError("topological mesh dimension must be 2 or 3")

    def advection_term(self, q):

        if self.state.mesh.topological_dimension() == 3:
            # <w,curl(u) cross ubar + grad( u.ubar)>
            # =<curl(u),ubar cross w> - <div(w), u.ubar>
            # =<u,curl(ubar cross w)> -
            #      <<u_upwind, [[n cross(ubar cross w)cross]]>>

            both = lambda u: 2*avg(u)

            L = (
                inner(q, curl(cross(self.ubar, self.test)))*dx
                - inner(both(self.Upwind*q),
                        both(cross(self.n, cross(self.ubar, self.test))))*self.dS
            )

        else:

            if self.ibp == "once":
                L = (
                    -inner(self.gradperp(inner(self.test, self.perp(self.ubar))), q)*dx
                    - inner(jump(inner(self.test, self.perp(self.ubar)), self.n),
                            self.perp_u_upwind(q))*self.dS
                )
            else:
                L = (
                    (-inner(self.test, div(self.perp(q))*self.perp(self.ubar)))*dx
                    - inner(jump(inner(self.test, self.perp(self.ubar)), self.n),
                            self.perp_u_upwind(q))*self.dS
                    + jump(inner(self.test,
                                 self.perp(self.ubar))*self.perp(q), self.n)*self.dS
                )

        L -= 0.5*div(self.test)*inner(q, self.ubar)*dx

        return L


class EulerPoincare(VectorInvariant):
    """
    Class defining the Euler-Poincare form of the vector advection equation.

    :arg state: :class:`.State` object.
    :arg V: Function space
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """

    def advection_term(self, q):
        L = super(EulerPoincare, self).advection_term(q)
        L -= 0.5*div(self.test)*inner(q, self.ubar)*dx
        return L
