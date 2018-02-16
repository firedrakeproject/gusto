from abc import ABCMeta, abstractmethod
from firedrake import Function, TestFunction, TrialFunction, \
    FacetNormal, \
    dx, dot, grad, div, jump, avg, dS, dS_v, dS_h, inner, \
    ds, ds_v, ds_t, ds_b, \
    outer, sign, cross, CellNormal, sqrt, Constant, \
    curl, BrokenElement, FunctionSpace
from gusto.configuration import DEBUG


__all__ = ["LinearAdvection", "AdvectionEquation", "EmbeddedDGAdvection", "SUPGAdvection", "VectorInvariant", "EulerPoincare"]


class TransportEquation(object, metaclass=ABCMeta):
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

    def __init__(self, state, V, *, ibp="once", solver_params=None):
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
            self.ds = None
        else:
            if V.extruded:
                self.dS = (dS_h + dS_v)
                self.ds = (ds_b + ds_t + ds_v)
            else:
                self.dS = dS
                self.ds = ds
            self.n = FacetNormal(state.mesh)
            self.un = 0.5*(dot(self.ubar, self.n) + abs(dot(self.ubar, self.n)))

        if solver_params:
            self.solver_parameters = solver_params

        # default solver options
        else:
            self.solver_parameters = {'ksp_type': 'cg',
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}
        if state.output.log_level == DEBUG:
            self.solver_parameters["ksp_monitor_true_residual"] = True

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
        super().__init__(state=state, V=V, ibp=ibp, solver_params=solver_params)
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
    :arg vector_manifold: Boolean. If true adds extra terms that are needed for
    advecting vector equations on manifolds.
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """
    def __init__(self, state, V, *, ibp="once", equation_form="advective",
                 vector_manifold=False, solver_params=None, outflow=False):
        super().__init__(state=state, V=V, ibp=ibp, solver_params=solver_params)
        if equation_form == "advective" or equation_form == "continuity":
            self.continuity = (equation_form == "continuity")
        else:
            raise ValueError("equation_form must be either 'advective' or 'continuity'")
        self.vector_manifold = vector_manifold
        self.outflow = outflow
        if outflow and ibp is None:
            raise ValueError("outflow is True and ibp is None are incompatible options")

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
                L -= (inner(self.test('+'),
                            dot(self.ubar('+'), self.n('+'))*q('+'))
                      + inner(self.test('-'),
                              dot(self.ubar('-'), self.n('-'))*q('-')))*self.dS

        if self.outflow:
            L += self.test*self.un*q*self.ds

        if self.vector_manifold:
            un = self.un
            w = self.test
            u = q
            n = self.n
            dS = self.dS
            L += un('+')*inner(w('-'), n('+')+n('-'))*inner(u('+'), n('+'))*dS
            L += un('-')*inner(w('+'), n('+')+n('-'))*inner(u('-'), n('-'))*dS
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
    :arg vector_manifold: Boolean. If true adds extra terms that are needed for
    advecting vector equations on manifolds.
    :arg Vdg: (optional) :class:`.FunctionSpace object. The embedding function
              space. Defaults to None which means that a broken space is
              constructed for you.
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    :arg recovered_spaces: A list or tuple of function spaces to be used for the
                           recovered space advection method. The must be three
                           spaces, indexed in the following order:
                           [0]: the embedded space in which the advection takes place.
                           [1]: the continuous recovered space.
                           [2]: a broken or discontinuous version of the original space.
                           The default for this option is None, in which case the method
                           will not be used.
    """

    def __init__(self, state, V, ibp="once", equation_form="advective", vector_manifold=False, Vdg=None, solver_params=None, recovered_spaces=None, outflow=False):

        # give equation the property V0, the space that the function should live in
        # in the absence of Vdg, this is used to set up the space for advection
        # to take place in
        self.V0 = V

        self.recovered = False
        if recovered_spaces is not None:
            # Vdg must be None to use recovered spaces
            if Vdg is not None:
                raise ValueError('The recovered_spaces option is incompatible with the Vdg option')
            else:
                # check that the list or tuple of spaces is the right length
                if len(recovered_spaces) != 3:
                    raise ValueError('recovered_spaces must be a list or tuple containing three spaces')
                self.space = recovered_spaces[0]  # the space in which advection happens
                self.V_rec = recovered_spaces[1]  # the recovered continuous space
                self.V_brok = recovered_spaces[2]  # broken version of V0
                self.recovered = True
        elif Vdg is None:
            # Create broken space, functions and projector
            V_elt = BrokenElement(V.ufl_element())
            self.space = FunctionSpace(state.mesh, V_elt)
        else:
            self.space = Vdg

        super().__init__(state=state,
                         V=self.space,
                         ibp=ibp,
                         equation_form=equation_form,
                         vector_manifold=vector_manifold,
                         solver_params=solver_params,
                         outflow=outflow)


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
    def __init__(self, state, V, ibp="twice", equation_form="advective", supg_params=None, solver_params=None, outflow=False):

        if not solver_params:
            # SUPG method leads to asymmetric matrix (since the test function
            # is effectively modified), so don't use CG
            solver_params = {'ksp_type': 'gmres',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}

        super().__init__(state=state, V=V, ibp=ibp,
                         equation_form=equation_form,
                         solver_params=solver_params,
                         outflow=outflow)

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
    def __init__(self, state, V, *, ibp="once", solver_params=None):
        super().__init__(state=state, V=V, ibp=ibp,
                         solver_params=solver_params)

        self.Upwind = 0.5*(sign(dot(self.ubar, self.n))+1)

        if self.state.mesh.topological_dimension() == 2:
            self.perp = state.perp
            if state.on_sphere:
                outward_normals = CellNormal(state.mesh)
                self.perp_u_upwind = lambda q: self.Upwind('+')*cross(outward_normals('+'), q('+')) + self.Upwind('-')*cross(outward_normals('-'), q('-'))
            else:
                self.perp_u_upwind = lambda q: self.Upwind('+')*state.perp(q('+')) + self.Upwind('-')*state.perp(q('-'))
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
        L = super().advection_term(q)
        L -= 0.5*div(self.test)*inner(q, self.ubar)*dx
        return L
