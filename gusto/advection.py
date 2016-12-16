from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function, LinearVariationalProblem, \
    LinearVariationalSolver, Projector
from gusto.transport_equation import LinearAdvection, EmbeddedDGAdvection


def embedded_dg(original_apply):
    """
    Decorator to add interpolation and projection steps for embedded
    DG advection.
    """
    def get_apply(self, x_in, x_out):
        if self.embedded_dg:
            def new_apply(self, x_in, x_out):
                self.xdg_in.interpolate(x_in)
                original_apply(self, self.xdg_in, self.xdg_out)
                self.Projector.project()
                x_out.assign(self.x_projected)
            return new_apply(self, x_in, x_out)

        else:
            return original_apply(self, x_in, x_out)
    return get_apply


class Advection(object):
    """
    Base class for advection schemes.

    :arg state: :class:`.State` object.
    :arg field: field to be advected
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg solver_params: solver_parameters
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, field, equation=None, solver_params=None):

        self.state = state
        self.field = field

        if solver_params is None:
            if isinstance(equation, LinearAdvection):
                self.solver_parameters = {'ksp_type':'cg',
                                          'pc_type':'bjacobi',
                                          'sub_pc_type': 'ilu'}
            else:
                self.solver_parameters = {'ksp_type':'preonly',
                                          'pc_type':'bjacobi',
                                          'sub_pc_type': 'ilu'}
        else:
            self.solver_parameters = solver_params
        self.dt = self.state.timestepping.dt

        # check to see if we are using an embedded DG method - is we are then
        # the projector and output function will have been set up in the
        # equation class and we can get the correct function space from
        # the output function.
        if isinstance(equation, EmbeddedDGAdvection):
            self.embedded_dg = True
            fs = equation.space
            self.xdg_in = Function(equation.space)
            self.xdg_out = Function(equation.space)
            self.x_projected = Function(field.function_space())
            parameters = {'ksp_type':'cg',
                          'pc_type':'bjacobi',
                          'sub_pc_type':'ilu'}
            self.Projector = Projector(self.xdg_out, self.x_projected,
                                       solver_parameters=parameters)
            self.xdg_in = Function(fs)
        else:
            self.embedded_dg = False
            fs = field.function_space()

        # setup required functions
        self.dq = Function(fs)
        self.q1 = Function(fs)

        # get ubar from the equation class if provided
        self.equation = equation
        if equation is not None:
            self.ubar = self.equation.ubar

    def update_ubar(self, xn, xnp1, alpha):
        un = xn.split()[0]
        unp1 = xnp1.split()[0]
        self.ubar.assign(un + alpha*(unp1-un))

    def update_solver(self):
        # setup solver using lhs and rhs defined in derived class

        problem = LinearVariationalProblem(self.lhs, self.rhs, self.dq)
        solver_name = self.field.name()+self.equation.__class__.__name__+self.__class__.__name__
        self.solver = LinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @abstractmethod
    def apply(self, x_in, x_out):
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
    pass


class NoAdvection(Advection):
    """
    An non-advection scheme that does nothing.
    """

    def update_ubar(self, xn, xnp1, alpha):
        pass

    def apply(self, x_in, x_out):

        x_out.assign(x_in)


class ForwardEuler(Advection):
    """
    Class to implement the forward Euler timestepping scheme:
    y_(n+1) = y_n + dt*L(y_n)
    where L is the advection operator
    """
    def __init__(self, state, field, equation, solver_params=None):
        super(ForwardEuler, self).__init__(state, field, equation, solver_params)

        self.lhs = self.equation.mass_term(self.equation.trial)
        self.rhs = -self.equation.advection_term(self.q1)
        self.update_solver()

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(x_in + self.dt*self.dq)


class SSPRK3(Advection):
    """
    Class to implement the Strongly Structure Preserving Runge Kutta 3-stage
    timestepping method:
    y^1 = y_n + L(y_n)
    y^2 = (3/4)y_n + (1/4)(y^1 + L(y^1))
    y_(n+1) = (1/3)y_n + (2/3)(y^2 + L(y^2))
    where subscripts indicate the timelevel, superscripts indicate the stage
    number and L is the advection operator.
    """
    def __init__(self, state, field, equation, solver_params=None):
        super(SSPRK3, self).__init__(state, field, equation, solver_params)

        self.lhs = self.equation.mass_term(self.equation.trial)
        self.rhs = self.equation.mass_term(self.q1) - self.dt*self.equation.advection_term(self.q1)
        self.update_solver()

    def solve_stage(self, x_in, stage):

        if stage == 0:
            self.solver.solve()
            self.q1.assign(self.dq)

        elif stage == 1:
            self.solver.solve()
            self.q1.assign(0.75*x_in + 0.25*self.dq)

        elif stage == 2:
            self.solver.solve()

    @embedded_dg
    def apply(self, x_in, x_out):

        self.q1.assign(x_in)
        for i in range(3):
            self.solve_stage(x_in, i)
        x_out.assign((1./3.)*x_in + (2./3.)*self.dq)


class ThetaMethod(Advection):
    """
    Class to implement the theta timestepping method:
    y_(n+1) = y_n + dt*(theta*L(y_n) + (1-theta)*L(y_(n+1))) where L is the advection operator.
    """
    def __init__(self, state, field, equation, theta=0.5, solver_params=None):
        super(ThetaMethod, self).__init__(state, field, equation, solver_params)
        trial = self.equation.trial
        self.lhs = self.equation.mass_term(trial) + theta*self.dt*self.equation.advection_term(trial)
        self.rhs = self.equation.mass_term(self.q1) - (1.-theta)*self.dt*self.equation.advection_term(self.q1)
        self.update_solver()

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

    def __init__(self, state, V, Vdg=None, continuity=False):

        if Vdg is None:
            Vdg_elt = BrokenElement(V.ufl_element())
            Vdg = FunctionSpace(state.mesh, Vdg_elt)

        super(EmbeddedDGAdvection, self).__init__(state, Vdg, continuity)

        self.xdg_in = Function(Vdg)
        self.xdg_out = Function(Vdg)

        self.x_projected = Function(V)
        pparameters = {'ksp_type':'cg',
                       'pc_type':'bjacobi',
                       'sub_pc_type':'ilu'}
        self.Projector = Projector(self.xdg_out, self.x_projected,
                                   solver_parameters=pparameters)

    def apply(self, x_in, x_out):

        self.xdg_in.interpolate(x_in)
        super(EmbeddedDGAdvection, self).apply(self.xdg_in, self.xdg_out)
        self.Projector.project()
        x_out.assign(self.x_projected)


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

        if state.mesh.geometric_dimension() == 3:

            if V.extruded:
                surface_measure = (dS_h + dS_v)
            else:
                surface_measure = dS

            # <w,curl(u) cross ubar + grad( u.ubar)>
            # =<curl(u),ubar cross w> - <div(w), u.ubar>
            # =<u,curl(ubar cross w)> -
            #      <<u_upwind, [[n cross(ubar cross w)cross]]>>

            both = lambda u: 2*avg(u)

            Eqn = (
                inner(w, u-self.u0)*dx
                + dt*inner(ustar, curl(cross(self.ubar, w)))*dx
                - dt*inner(both(Upwind*ustar),
                           both(cross(n, cross(self.ubar, w))))*surface_measure
                - dt*div(w)*inner(ustar, self.ubar)*dx
            )

        # define surface measure and terms involving perp differently
        # for slice (i.e. if V.extruded is True) and shallow water
        # (V.extruded is False)
        else:
            if state.on_sphere:
                surface_measure = dS
                outward_normals = CellNormal(state.mesh)
                perp = lambda u: cross(outward_normals, u)
                perp_u_upwind = Upwind('+')*cross(outward_normals('+'),ustar('+')) + Upwind('-')*cross(outward_normals('-'),ustar('-'))
            else:
                perp = lambda u: as_vector([-u[1], u[0]])
                perp_u_upwind = Upwind('+')*perp(ustar('+')) + Upwind('-')*perp(ustar('-'))
                if V.extruded:
                    surface_measure = (dS_h + dS_v)
                else:
                    surface_measure = dS

            Eqn = (
                (inner(w, u-self.u0)
                 - dt*inner(w, div(perp(ustar))*perp(self.ubar))
                 - dt*div(w)*inner(ustar, self.ubar))*dx
                - dt*inner(jump(inner(w, perp(self.ubar)), n),
                           perp_u_upwind)*surface_measure
                + dt*jump(inner(w,
                                perp(self.ubar))*perp(ustar), n)*surface_measure
            )

        a = lhs(Eqn)
        L = rhs(Eqn)
        self.u1 = Function(V)
        uproblem = LinearVariationalProblem(a, L, self.u1)
        self.usolver = LinearVariationalSolver(uproblem,
                                               options_prefix='EPAdvection')

    def apply(self, x_in, x_out):
        self.u0.assign(x_in)
        self.usolver.solve()
        x_out.assign(self.u1)


class SUPGAdvection(Advection):
    """
    An SUPG advection scheme that can apply DG upwinding (in the direction
    specified by the direction arg) if the function space is only
    partially continuous.

    :arg state: :class:`.State` object.
    :arg V:class:`.FunctionSpace` object. The advected field function space.
    :arg direction: list containing the directions in which the function
    space is discontinuous. 1 corresponds to the vertical direction, 2 to
    the horizontal direction
    :arg supg_params: dictionary containing SUPG parameters tau for each
    direction. If not supplied tau is set to dt/sqrt(15.)
    """
    def __init__(self, state, V, direction=[], supg_params=None):
        super(SUPGAdvection, self).__init__(state)
        dt = state.timestepping.dt
        params = supg_params.copy() if supg_params else {}
        params.setdefault('a0', dt/sqrt(15.))
        params.setdefault('a1', dt/sqrt(15.))

        gamma = TestFunction(V)
        theta = TrialFunction(V)
        self.theta0 = Function(V)

        # make SUPG test function
        taus = [params["a0"], params["a1"]]
        for i in direction:
            taus[i] = 0.0
        tau = Constant(((taus[0], 0.), (0., taus[1])))

        dgamma = dot(dot(self.ubar, tau), grad(gamma))
        gammaSU = gamma + dgamma

        n = FacetNormal(state.mesh)
        un = 0.5*(dot(self.ubar, n) + abs(dot(self.ubar, n)))

        a_mass = gammaSU*theta*dx
        arhs = a_mass - dt*gammaSU*dot(self.ubar, grad(theta))*dx

        if 1 in direction:
            arhs -= (
                dt*dot(jump(gammaSU), (un('+')*theta('+')
                                       - un('-')*theta('-')))*dS_v
                - dt*(gammaSU('+')*dot(self.ubar('+'), n('+'))*theta('+')
                      + gammaSU('-')*dot(self.ubar('-'), n('-'))*theta('-'))*dS_v
            )
        if 2 in direction:
            arhs -= (
                dt*dot(jump(gammaSU), (un('+')*theta('+')
                                       - un('-')*theta('-')))*dS_h
                - dt*(gammaSU('+')*dot(self.ubar('+'), n('+'))*theta('+')
                      + gammaSU('-')*dot(self.ubar('-'), n('-'))*theta('-'))*dS_h
            )

        self.theta1 = Function(V)
        self.dtheta = Function(V)
        problem = LinearVariationalProblem(a_mass, action(arhs,self.theta1), self.dtheta)
        self.solver = LinearVariationalSolver(problem,
                                              options_prefix='SUPGAdvection')

    def apply(self, x_in, x_out):

        # SSPRK Stage 1
        self.theta1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)
