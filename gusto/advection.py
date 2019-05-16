from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import (Function, LinearVariationalProblem, Constant,
                       LinearVariationalSolver, Projector, Interpolator,
                       TrialFunction, TestFunction, inner, dx, lhs, rhs,
                       DirichletBC, grad)
from firedrake.utils import cached_property
from gusto.configuration import logger, DEBUG
from gusto.recovery import Recoverer


__all__ = ["NoAdvection", "ForwardEuler", "SSPRK3", "ThetaMethod",
           "Update_advection", "Reconstruct_q"]


def embedded_dg(original_apply):
    """
    Decorator to add interpolation and projection steps for embedded
    DG advection.
    """
    def get_apply(self, x_in, x_out):

        if self.discretisation_option in ["embedded_dg", "recovered"]:

            def new_apply(self, x_in, x_out):

                self.pre_apply(x_in, self.discretisation_option)
                original_apply(self, self.xdg_in, self.xdg_out)
                self.post_apply(x_out, self.discretisation_option)

            return new_apply(self, x_in, x_out)

        else:

            return original_apply(self, x_in, x_out)

    return get_apply


class Advection(object, metaclass=ABCMeta):
    """
    Base class for advection schemes.

    :arg state: :class:`.State` object.
    :arg field: field to be advected
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg solver_parameters: solver_parameters
    :arg limiter: :class:`.Limiter` object.
    :arg options: :class:`.AdvectionOptions` object
    """

    def __init__(self, state, field, equation=None, *, solver_parameters=None,
                 limiter=None):

        if equation is not None:

            self.state = state
            self.field = field
            self.equation = equation
            self.dt = self.state.timestepping.dt

            # get default solver options if none passed in
            if solver_parameters is None:
                self.solver_parameters = equation.solver_parameters
            else:
                self.solver_parameters = solver_parameters
                if logger.isEnabledFor(DEBUG):
                    self.solver_parameters["ksp_monitor_true_residual"] = True

            self.limiter = limiter

            if hasattr(equation, "options"):
                self.discretisation_option = equation.options.name
                self._setup(state, field, equation.options)
            else:
                self.discretisation_option = None
                self.fs = field.function_space()

            # setup required functions
            self.dq = Function(self.fs)
            self.q1 = Function(self.fs)

    def _setup(self, state, field, options):

        if options.name in ["embedded_dg", "recovered"]:
            self.fs = options.embedding_space
            self.xdg_in = Function(self.fs)
            self.xdg_out = Function(self.fs)
            self.x_projected = Function(field.function_space())
            parameters = {'ksp_type': 'cg',
                          'pc_type': 'bjacobi',
                          'sub_pc_type': 'ilu'}
            self.Projector = Projector(self.xdg_out, self.x_projected,
                                       solver_parameters=parameters)

        if options.name == "recovered":
            # set up the necessary functions
            self.x_in = Function(field.function_space())
            x_rec = Function(options.recovered_space)
            x_brok = Function(options.broken_space)

            # set up interpolators and projectors
            self.x_rec_projector = Recoverer(self.x_in, x_rec, VDG=self.fs, boundary_method=options.boundary_method)  # recovered function
            self.x_brok_projector = Projector(x_rec, x_brok)  # function projected back
            self.xdg_interpolator = Interpolator(self.x_in + x_rec - x_brok, self.xdg_in)
            if self.limiter is not None:
                self.x_brok_interpolator = Interpolator(self.xdg_out, x_brok)
                self.x_out_projector = Recoverer(x_brok, self.x_projected)

    def pre_apply(self, x_in, discretisation_option):
        """
        Extra steps to advection if using an embedded method,
        which might be either the plain embedded method or the
        recovered space advection scheme.

        :arg x_in: the input set of prognostic fields.
        :arg discretisation option: string specifying which scheme to use.
        """
        if discretisation_option == "embedded_dg":
            try:
                self.xdg_in.interpolate(x_in)
            except NotImplementedError:
                self.xdg_in.project(x_in)

        elif discretisation_option == "recovered":
            self.x_in.assign(x_in)
            self.x_rec_projector.project()
            self.x_brok_projector.project()
            self.xdg_interpolator.interpolate()

    def post_apply(self, x_out, discretisation_option):
        """
        The projection steps, returning a field to its original space
        for an embedded DG advection scheme. For the case of the
        recovered scheme, there are two options dependent on whether
        the scheme is limited or not.

        :arg x_out: the outgoing field.
        :arg discretisation_option: string specifying which option to use.
        """
        if discretisation_option == "embedded_dg":
            self.Projector.project()

        elif discretisation_option == "recovered":
            if self.limiter is not None:
                self.x_brok_interpolator.interpolate()
                self.x_out_projector.project()
            else:
                self.Projector.project()
        x_out.assign(self.x_projected)

    @abstractproperty
    def lhs(self):
        return self.equation.mass_term(self.equation.trial)

    @abstractproperty
    def rhs(self):
        return self.equation.mass_term(self.q1) - self.dt*self.equation.advection_term(self.q1)

    @cached_property
    def solver(self):
        # setup solver using lhs and rhs defined in derived class
        problem = LinearVariationalProblem(self.lhs, self.rhs, self.dq)
        solver_name = self.field.name()+self.equation.__class__.__name__+self.__class__.__name__
        return LinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

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

    def lhs(self):
        pass

    def rhs(self):
        pass

    def apply(self, x_in, x_out):
        x_out.assign(x_in)


class ExplicitAdvection(Advection):
    """
    Base class for explicit advection schemes.

    :arg state: :class:`.State` object.
    :arg field: field to be advected
    :arg equation: :class:`.Equation` object, specifying the equation
    that field satisfies
    :arg subcycles: (optional) integer specifying number of subcycles to perform
    :arg solver_parameters: solver_parameters
    :arg limiter: :class:`.Limiter` object.
    """

    def __init__(self, state, field, equation=None, *, subcycles=None,
                 solver_parameters=None, limiter=None):
        super().__init__(state, field, equation,
                         solver_parameters=solver_parameters, limiter=limiter)

        # if user has specified a number of subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
        if subcycles is not None:
            self.dt = self.dt/subcycles
            self.ncycles = subcycles
        else:
            self.dt = self.dt
            self.ncycles = 1
        self.x = [Function(self.fs)]*(self.ncycles+1)

    @abstractmethod
    def apply_cycle(self, x_in, x_out):
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass

    @embedded_dg
    def apply(self, x_in, x_out):
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        self.x[0].assign(x_in)
        for i in range(self.ncycles):
            self.apply_cycle(self.x[i], self.x[i+1])
            self.x[i].assign(self.x[i+1])
        x_out.assign(self.x[self.ncycles-1])


class ForwardEuler(ExplicitAdvection):
    """
    Class to implement the forward Euler timestepping scheme:
    y_(n+1) = y_n + dt*L(y_n)
    where L is the advection operator
    """

    @cached_property
    def lhs(self):
        return super(ForwardEuler, self).lhs

    @cached_property
    def rhs(self):
        return super(ForwardEuler, self).rhs

    def apply_cycle(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class SSPRK3(ExplicitAdvection):
    """
    Class to implement the Strongly Structure Preserving Runge Kutta 3-stage
    timestepping method:
    y^1 = y_n + L(y_n)
    y^2 = (3/4)y_n + (1/4)(y^1 + L(y^1))
    y_(n+1) = (1/3)y_n + (2/3)(y^2 + L(y^2))
    where subscripts indicate the timelevel, superscripts indicate the stage
    number and L is the advection operator.
    """

    @cached_property
    def lhs(self):
        return super(SSPRK3, self).lhs

    @cached_property
    def rhs(self):
        return super(SSPRK3, self).rhs

    def solve_stage(self, x_in, stage):

        if stage == 0:
            self.solver.solve()
            self.q1.assign(self.dq)

        elif stage == 1:
            self.solver.solve()
            self.q1.assign(0.75*x_in + 0.25*self.dq)

        elif stage == 2:
            self.solver.solve()
            self.q1.assign((1./3.)*x_in + (2./3.)*self.dq)

        if self.limiter is not None:
            self.limiter.apply(self.q1)

    def apply_cycle(self, x_in, x_out):

        if self.limiter is not None:
            self.limiter.apply(x_in)

        self.q1.assign(x_in)
        for i in range(3):
            self.solve_stage(x_in, i)
        x_out.assign(self.q1)


class ThetaMethod(Advection):
    """
    Class to implement the theta timestepping method:
    y_(n+1) = y_n + dt*(theta*L(y_n) + (1-theta)*L(y_(n+1))) where L is the advection operator.
    arg weight: Either name of prognostic variable z (for weights z_(n), z_(n+1)),
    or 2-tuple containing weight for y_(n+1) and y_(n) respectively. Defaults to None.
    """
    def __init__(self, state, field, equation, theta=0.5, weight=None, solver_parameters=None):
        if weight is not None:
            if weight in state.fieldlist:
                self.weight_n = state.xn.split()[state.fieldlist.index(weight)]
                self.weight_p = state.xp.split()[state.fieldlist.index(weight)]
            elif len(weight) == 2:
                self.weight_n = weight[0]
                self.weight_p = weight[1]
            else:
                raise ValueError("weight should be a prognostic variable or a 2-tuple")
        else:
            self.weight_n = Constant(1.)
            self.weight_p = Constant(1.)

        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super(ThetaMethod, self).__init__(state, field, equation,
                                          solver_parameters=solver_parameters)

        self.theta = theta

    @cached_property
    def lhs(self):
        eqn = self.equation
        trial = eqn.trial
        return eqn.mass_term(self.weight_p*trial) + self.theta*self.dt*eqn.advection_term(self.state.h_project(trial))

    @cached_property
    def rhs(self):
        eqn = self.equation
        return eqn.mass_term(self.weight_n*self.q1) - (1.-self.theta)*self.dt*eqn.advection_term(self.state.h_project(self.q1))

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class Update_advection(object):
    """
    Class to update fields related to advection, such as ubar and flux
    time averages. Contains solvers for flux and flux-recovered velocity."""
    def __init__(self, state, active_advection=None):
        self.state = state

        self.Vu = state.spaces("HDiv")
        if self.Vu.extruded:
            self.bcs = [DirichletBC(self.Vu, 0.0, "bottom"),
                        DirichletBC(self.Vu, 0.0, "top")]
        else:
            self.bcs = None

        # Figure out which solvers to use
        if active_advection is not None:
            flux_forms = ['q' in state.fieldlist]
            for _, advection in active_advection:
                try:
                    flux_forms.append(advection.equation.flux_form)
                except AttributeError:
                    continue
            self.get_flux = any(flux_forms)
            if self.get_flux:
                self._setup_flux_solver(state)
            self.get_u_rec = False
            if state.hamiltonian:
                if state.hamiltonian_options.no_u_rec:
                    self._setup_flux_solver(state)
                    self.get_flux = True
                else:
                    self._setup_u_rec_solver(state)
                    self.get_u_rec = True
        else:
            self.get_flux = False
            self.get_u_rec = False

        # Use counter to avoid using solvers more than once per iteration
        self.update_count = 0
        if active_advection is not None:
            self.total_updates = len(active_advection)
        else:
            self.nr_of_updates = 1
        self.flux_updated = False
        self.u_rec_updated = False

    def _setup_flux_solver(self, state):
        # flux from u, d
        u_ = TrialFunction(self.Vu)
        v = TestFunction(self.Vu)
        un, dn = state.xn.split()[:2]
        self.F = Function(self.Vu)
        unp1, dnp1 = state.xnp1.split()[:2]
        Frhs = unp1*dnp1/3. + un*dnp1/6. + unp1*dn/6. + un*dn/3.
        F_eqn = inner(v, u_ - Frhs)*dx
        F_problem = LinearVariationalProblem(lhs(F_eqn), rhs(F_eqn), self.F, bcs=self.bcs)
        self.F_solver = LinearVariationalSolver(F_problem,
                                                solver_parameters={"ksp_type": "preonly",
                                                                   "pc_type": "lu"})

    def _setup_u_rec_solver(self, state):
        # u recovery from flux
        u_ = TrialFunction(self.Vu)
        v = TestFunction(self.Vu)
        self.u_rec = Function(self.Vu)
        un, dn = state.xn.split()[:2]
        unp1, dnp1 = state.xnp1.split()[:2]
        Frhs = unp1*dnp1/3. + un*dnp1/6. + unp1*dn/6. + un*dn/3.
        u_rec_eqn = inner(v, 0.5*(dn + dnp1)*u_ - Frhs)*dx
        u_rec_problem = LinearVariationalProblem(lhs(u_rec_eqn), rhs(u_rec_eqn),
                                                 self.u_rec, bcs=self.bcs)
        self.u_rec_solver = LinearVariationalSolver(u_rec_problem,
                                                    solver_parameters={"ksp_type": "preonly",
                                                                       "pc_type": "lu"})

    def apply(self, xn, xnp1, alpha, passive=False):
        un = xn.split()[0]
        unp1 = xnp1.split()[0]
        if passive:
            if self.state.hamiltonian:
                if not self.state.hamiltonian_options.no_u_rec:
                    self.state.u_rec.assign(0.5*(un + unp1))
            else:
                self.state.ubar.assign(un + alpha*(unp1-un))
        else:
            if self.get_flux and not self.flux_updated:
                self.F_solver.solve()
                self.state.F.assign(self.F)
                self.flux_updated = True
            if self.get_u_rec and not self.u_rec_updated:
                self.u_rec_solver.solve()
                self.state.ubar.assign(self.u_rec)
                self.state.u_rec.assign(self.u_rec)
                self.u_rec_updated = True
            if self.state.hamiltonian:
                self.state.upbar.assign(0.5*(un + unp1))
                if self.state.hamiltonian_options.no_u_rec:
                    self.state.ubar.assign(0.5*(un + unp1))
            else:
                self.state.ubar.assign(un + alpha*(unp1-un))

            self.update_count += 1
            # At the end of active advection loop, reset counter and solver flags
            if self.update_count == self.total_updates:
                self.update_count = 0
                self.flux_updated = False
                self.u_rec_updated = False


class Reconstruct_q(object):
    """
    Class to reconstruct vorticity q given velocity field u. Used
    when vorticity and velocity are advected independently, to avoid
    decoupling of the two fields.
    :arg state: :class:`.State` object.
    """
    def __init__(self, state):
        self.state = state
        self.x0 = Function(state.W)
        self.q = self.x0.split()[-1]
        self.Vq = state.spaces("Vq")

        self.lu_params = {"ksp_type": "preonly",
                          "pc_type": "lu"}

        if self.Vq.extruded:
            # If extruded, we need to get Zprime, the vorticity along the boundary
            self.Z = Function(self.Vq)
            self.Zring = Function(self.Vq)
            self.Zprime = Function(self.Vq)
            self._setup_reconstruction_solvers_extruded()
        else:
            self._setup_reconstruction_solver()

    def _setup_reconstruction_solver(self):
        gamma = TestFunction(self.Vq)
        q = TrialFunction(self.Vq)
        un, Dn = self.x0.split()[:2]

        q_recon_eqn = gamma*q*Dn*dx + inner(self.state.perp(grad(gamma)), un)*dx
        if hasattr(self.state.fields, "coriolis"):
            f = self.state.fields("coriolis")
            q_recon_eqn -= gamma*f*dx

        q_recon_problem = LinearVariationalProblem(lhs(q_recon_eqn), rhs(q_recon_eqn),
                                                   self.q)
        self.q_recon_solver = LinearVariationalSolver(q_recon_problem,
                                                      solver_parameters=self.lu_params)

    def _setup_reconstruction_solvers_extruded(self):
        un, Dn = self.x0.split()[:2]

        bcs = [DirichletBC(self.Vq, 0., "bottom"),
               DirichletBC(self.Vq, 0., "top")]

        self.qD_to_Z_Projector = Projector(self.q*Dn, self.Z, solver_parameters=self.lu_params)
        self.Z_to_Zring_Projector = Projector(self.Z, self.Zring, bcs=bcs,
                                              solver_parameters=self.lu_params)
        gamma = TestFunction(self.Vq)
        Z = TrialFunction(self.Vq)
        u_Zringeqn = gamma*Z*dx + inner(self.state.perp(grad(gamma)), un)*dx
        if hasattr(self.state.fields, "coriolis"):
            f = self.state.fields("coriolis")
            u_Zringeqn -= gamma*f*dx

        u_Zringproblem = LinearVariationalProblem(lhs(u_Zringeqn), rhs(u_Zringeqn),
                                                  self.Zring, bcs=bcs)
        self.u_Zring_solver = LinearVariationalSolver(u_Zringproblem,
                                                      solver_parameters=self.lu_params)
        q_recon_eqn = gamma*(Dn*Z - self.Z)*dx
        q_recon_problem = LinearVariationalProblem(lhs(q_recon_eqn), rhs(q_recon_eqn),
                                                   self.q)
        self.q_recon_solver = LinearVariationalSolver(q_recon_problem,
                                                      solver_parameters=self.lu_params)

    def apply(self, x_in):
        self.x0.assign(x_in)
        if self.Vq.extruded:
            # get Zprime from q
            self.qD_to_Z_Projector.project()
            self.Z_to_Zring_Projector.project()
            self.Zprime.assign(self.Z - self.Zring)
            # get Zring from u
            self.u_Zring_solver.solve()
            # Z = Zprime + Zring
            self.Z.assign(self.Zprime + self.Zring)
        self.q_recon_solver.solve()

        q_recon = x_in.split()[-1]
        q_recon.assign(self.q)
