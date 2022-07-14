from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import (Function, assemble, DirichletBC,
                       NonlinearVariationalProblem,
                       NonlinearVariationalSolver, Projector, Interpolator,
                       BrokenElement, VectorElement, FunctionSpace, split,
                       TestFunction, Constant, dot, grad, as_ufl)
from firedrake.formmanipulation import split_form
from firedrake.utils import cached_property
import ufl
from gusto.configuration import logger, DEBUG
from gusto.labels import (time_derivative, advecting_velocity, prognostic,
                          replace_subject, replace_test_function, implicit,
                          explicit, advection, subject)
from gusto.recovery import Recoverer
from gusto.fml.form_manipulation_labelling import Term, all_terms, drop
import numpy as np
import scipy
from scipy.special import legendre


__all__ = ["ForwardEuler", "BackwardEuler", "IMEX_Euler", "SSPRK3", "ThetaMethod", "ImplicitMidpoint", "FE_SDC", "BE_SDC", "IMEX_SDC"]


def is_cg(V):
    """
    Function to check if a given space, V, is CG. Broken elements are
    always discontinuous; for vector elements we check the names of
    the sobolev spaces of the subelements and for all other elements
    we just check the sobolev space name.
    """
    ele = V.ufl_element()
    if isinstance(ele, BrokenElement):
        return False
    elif type(ele) == VectorElement:
        return all([e.sobolev_space().name == "H1" for e in ele._sub_elements])
    else:
        return V.ufl_element().sobolev_space().name == "H1"


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

    def __init__(self, state, field_name=None, solver_parameters=None,
                 limiter=None, options=None):

        self.state = state
        self.field_name = field_name

        self.dt = Constant(self.state.dt)

        self.limiter = limiter

        self.options = options
        if options is not None:
            self.discretisation_option = options.name
        else:
            self.discretisation_option = None

        # get default solver options if none passed in
        if solver_parameters is None:
            self.solver_parameters = {'ksp_type': 'cg',
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}
        else:
            self.solver_parameters = solver_parameters
            if logger.isEnabledFor(DEBUG):
                self.solver_parameters["ksp_monitor_true_residual"] = None

    def setup(self, equation, uadv=None, apply_bcs=True, *active_labels, **kwargs):
        if "residual" not in kwargs.keys():
            self.residual = equation.residual
        else:
            self.residual = kwargs.get("residual")

        if self.field_name is not None:
            self.idx = equation.field_names.index(self.field_name)
            self.fs = self.state.fields(self.field_name).function_space()
            self.residual = self.residual.label_map(
                lambda t: t.get(prognostic) == self.field_name,
                lambda t: Term(
                    split_form(t.form)[self.idx].form,
                    t.labels),
                drop)
            bcs = equation.bcs[self.field_name]
        else:
            self.field_name = equation.field_name
            self.fs = equation.function_space
            if len(self.fs) > 1:
                bcs = []
                for k, v in equation.bcs.items():
                    idx = equation.field_names.index(k)
                    bcs += [DirichletBC(self.fs.sub(idx), bc.function_arg, bc.sub_domain) for bc in v]
            else:
                bcs = equation.bcs[self.field_name]
            self.idx = None
        if apply_bcs:
            self.bcs = bcs
        else:
            self.bcs = None

        if len(active_labels) > 0:
            self.residual = self.residual.label_map(
                lambda t: any(t.has_label(time_derivative, *active_labels)),
                map_if_false=drop)

        options = self.options

        self.replace_advecting_velocity(uadv)

        if self.discretisation_option in ["embedded_dg", "recovered"]:
            # construct the embedding space if not specified
            if options.embedding_space is None:
                V_elt = BrokenElement(self.fs.ufl_element())
                self.fs = FunctionSpace(self.state.mesh, V_elt)
            else:
                self.fs = options.embedding_space
            self.xdg_in = Function(self.fs)
            self.xdg_out = Function(self.fs)
            if self.idx is None:
                self.x_projected = Function(equation.function_space)
            else:
                self.x_projected = Function(self.state.fields(self.field_name).function_space())
            new_test = TestFunction(self.fs)
            parameters = {'ksp_type': 'cg',
                          'pc_type': 'bjacobi',
                          'sub_pc_type': 'ilu'}

        if self.discretisation_option == "supg":
            # construct tau, if it is not specified
            dim = self.state.mesh.topological_dimension()
            if options.tau is not None:
                # if tau is provided, check that is has the right size
                tau = options.tau
                assert as_ufl(tau).ufl_shape == (dim, dim), "Provided tau has incorrect shape!"
            else:
                # create tuple of default values of size dim
                default_vals = [options.default*self.dt]*dim
                # check for directions is which the space is discontinuous
                # so that we don't apply supg in that direction
                if is_cg(self.fs):
                    vals = default_vals
                else:
                    space = self.fs.ufl_element().sobolev_space()
                    if space.name in ["HDiv", "DirectionalH"]:
                        vals = [default_vals[i] if space[i].name == "H1"
                                else 0. for i in range(dim)]
                    else:
                        raise ValueError("I don't know what to do with space %s" % space)
                tau = Constant(tuple([
                    tuple(
                        [vals[j] if i == j else 0. for i, v in enumerate(vals)]
                    ) for j in range(dim)])
                )
                self.solver_parameters = {'ksp_type': 'gmres',
                                          'pc_type': 'bjacobi',
                                          'sub_pc_type': 'ilu'}

            test = TestFunction(self.fs)
            new_test = test + dot(dot(uadv, tau), grad(test))

        if self.discretisation_option is not None:
            # replace the original test function with one defined on
            # the embedding space, as this is the space where the
            # advection occurs
            self.residual = self.residual.label_map(
                all_terms,
                map_if_true=replace_test_function(new_test))

        if self.discretisation_option == "embedded_dg":
            if self.limiter is None:
                self.x_out_projector = Projector(self.xdg_out, self.x_projected,
                                                 solver_parameters=parameters)
            else:
                self.x_out_projector = Recoverer(self.xdg_out, self.x_projected)

        if self.discretisation_option == "recovered":
            # set up the necessary functions
            self.x_in = Function(self.state.fields(self.field_name).function_space())
            x_rec = Function(options.recovered_space)
            x_brok = Function(options.broken_space)

            # set up interpolators and projectors
            self.x_rec_projector = Recoverer(self.x_in, x_rec, VDG=self.fs, boundary_method=options.boundary_method)  # recovered function
            self.x_brok_projector = Projector(x_rec, x_brok)  # function projected back
            self.xdg_interpolator = Interpolator(self.x_in + x_rec - x_brok, self.xdg_in)
            if self.limiter is not None:
                self.x_brok_interpolator = Interpolator(self.xdg_out, x_brok)
                self.x_out_projector = Recoverer(x_brok, self.x_projected)
            else:
                self.x_out_projector = Projector(self.xdg_out, self.x_projected)

        # setup required functions
        self.dq = Function(self.fs)
        self.q1 = Function(self.fs)

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
        if discretisation_option == "recovered" and self.limiter is not None:
            self.x_brok_interpolator.interpolate()
        self.x_out_projector.project()
        x_out.assign(self.x_projected)

    @abstractproperty
    def lhs(self):
        l = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.dq, self.idx),
            map_if_false=drop)

        return l.form

    @abstractproperty
    def rhs(self):
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.q1, self.idx))

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: -self.dt*t)

        return r.form

    def replace_advecting_velocity(self, uadv=None):

        # replace the advecting velocity in any terms that contain it
        if any([t.has_label(advecting_velocity) for t in self.residual]):
            if uadv is not None:
                self.residual = self.residual.label_map(
                    lambda t: t.has_label(advecting_velocity),
                    map_if_true=lambda t: Term(ufl.replace(
                        t.form, {t.get(advecting_velocity): uadv}), t.labels)
                )
                self.residual = advecting_velocity.update_value(self.residual, uadv)
            else:
                # assumes explicit
                if any([t.has_label(advecting_velocity) for t in self.residual]):
                    self.residual = self.residual.label_map(
                        lambda t: t.has_label(advecting_velocity),
                        map_if_true=lambda t: Term(ufl.replace(
                            t.form, {t.get(advecting_velocity):
                                     split(t.get(subject))[0]}), t.labels)
                    )
                    self.residual = advecting_velocity.remove(self.residual)

    @cached_property
    def solver(self):
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs-self.rhs, self.dq, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, solver_parameters=self.solver_parameters, options_prefix=solver_name)

    @abstractmethod
    def apply(self, x_in, x_out):
        """
        Function takes x as input, computes L(x) as defined by the equation,
        and returns x_out as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass


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

    def __init__(self, state, field_name=None, subcycles=None,
                 solver_parameters=None, limiter=None, options=None):
        super().__init__(state, field_name,
                         solver_parameters=solver_parameters,
                         limiter=limiter, options=options)

        self.subcycles = subcycles

    def setup(self, equation, uadv=None, *active_labels, **kwargs):

        super().setup(equation, uadv, *active_labels, **kwargs)

        # if user has specified a number of subcycles, then save this
        # and rescale dt accordingly; else perform just one cycle using dt
        if self.subcycles is not None:
            self.dt = self.dt/self.subcycles
            self.ncycles = self.subcycles
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


class BackwardEuler(Advection):

    @property
    def lhs(self):
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.dq, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.dt*t)

        return l.form

    @property
    def rhs(self):

        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(self.q1, self.idx),
            map_if_false=drop)

        return r.form

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class IMEX_Euler(Advection):

    @property
    def lhs(self):
        l = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: self.dt*t)

        l = l.label_map(
            lambda t: any(t.has_label(implicit, time_derivative)),
            replace_subject(self.dq),
            drop
        )
        return l.form

    @property
    def rhs(self):
        r = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=lambda t: self.dt*t)

        r = r.label_map(
            lambda t: any(t.has_label(explicit, time_derivative)),
            replace_subject(self.q1.split()),
            drop
        )

        r = r.label_map(
            lambda t: t.has_label(time_derivative),
            lambda t: -1*t
        )
        return r.form

    @cached_property
    def solver(self):
        # setup solver using lhs and rhs defined in derived class
        problem = NonlinearVariationalProblem(self.lhs + self.rhs, self.dq, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        return NonlinearVariationalSolver(problem, options_prefix=solver_name)

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class ThetaMethod(Advection):
    """
    Class to implement the theta timestepping method:
    y_(n+1) = y_n + dt*(theta*L(y_n) + (1-theta)*L(y_(n+1))) where L is the advection operator.
    """
    def __init__(self, state, field_name=None, theta=None,
                 solver_parameters=None, options=None):

        if theta is None:
            raise ValueError("please provide a value for theta between 0 and 1")
        if not solver_parameters:
            # theta method leads to asymmetric matrix, per lhs function below,
            # so don't use CG
            solver_parameters = {'ksp_type': 'gmres',
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'}

        super().__init__(state, field_name,
                         solver_parameters=solver_parameters,
                         options=options)

        self.theta = theta

    @cached_property
    def lhs(self):
        l = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.dq, self.idx))
        l = l.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: self.theta*self.dt*t)

        return l.form

    @cached_property
    def rhs(self):
        r = self.residual.label_map(
            all_terms,
            map_if_true=replace_subject(self.q1, self.idx))
        r = r.label_map(lambda t: t.has_label(time_derivative),
                        map_if_false=lambda t: -(1-self.theta)*self.dt*t)

        return r.form

    def apply(self, x_in, x_out):
        self.q1.assign(x_in)
        self.solver.solve()
        x_out.assign(self.dq)


class ImplicitMidpoint(ThetaMethod):
    """
    Class to implement the implicit midpoint timestepping method, i.e. the
    theta method with theta=0.5:
    y_(n+1) = y_n + 0.5*dt*(L(y_n) + L(y_(n+1)))
    where L is the advection operator.
    """
    def __init__(self, state, field_name=None, solver_parameters=None,
                 options=None):
        super().__init__(state, field_name, theta=0.5,
                         solver_parameters=solver_parameters,
                         options=options)


class SDC(object, metaclass=ABCMeta):

    def __init__(self, state, M, maxk):

        self.state = state
        self.dt = Constant(state.dt)
        self.M = M
        self.maxk = maxk

        self.rnw_r(state.dt)
        self.Qmatrix()
        self.Smatrix()
        self.dtau = np.diff(np.append(0, self.nodes))

    @abstractmethod
    def setup(self, equation, uadv=None):
        pass

    def rnw_r(self, b, A=-1, B=1):
        # nodes and weights for gauss - radau IIA quadrature
        # See Abramowitz & Stegun p 888
        M = self.M
        a = 0
        nodes = np.zeros(M)
        nodes[0] = A
        p = np.poly1d([1, 1])
        pn = legendre(M)
        pn1 = legendre(M-1)
        poly, remainder = (pn + pn1)/p  # [1] returns remainder from polynomial division
        nodes[1:] = np.sort(poly.roots)
        weights = 1/M**2 * (1-nodes[1:])/(pn1(nodes[1:]))**2
        weights = np.append(2/M**2, weights)
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights = (b - a)/(B - A)*weights
        self.nodes = ((b + a) - nodes)[::-1]  # reverse nodes
        self.weights = weights[::-1]  # reverse weights

    def NewtonVM(self, t):
        """
        t: array or list containing nodes.
        returns: array Newton Vandermode Matrix. Entries are in the lower
        triangle
        Polynomial can be created with
        scipy.linalg.solve_triangular(NewtonVM(t),y,lower=True) where y
        contains the points the polynomial need to pass through
        """
        t = np.asarray(t)
        dim = len(t)
        VM = np.zeros([dim, dim])
        VM[:, 0] = 1
        for i in range(1, dim):
            VM[:, i] = (t[:] - t[(i - 1)]) * VM[:, i - 1]

        return VM

    def Horner_newton(self, weights, xi, x):
        """
        Horner scheme to evaluate polynomials based on newton basis
        """
        y = np.zeros_like(x)
        for i in range(len(weights)):
            y = y * (x - xi[(-i - 1)]) + weights[(-i - 1)]

        return y

    def gauss_legendre(self, n, b, A=-1, B=1):
        # nodes and weights for gauss legendre quadrature
        a = 0
        poly = legendre(n)
        polyd = poly.deriv()
        nodes = poly.roots
        nodes = np.sort(nodes)
        weights = 2/((1-nodes**2)*(np.polyval(polyd, nodes))**2)
        gl_nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        gl_weights = (b-a)/(B-A)*weights
        return gl_nodes, gl_weights

    def get_weights(self, b):
        # This calculates for equation 2.4 FWSW - called from Q
        # integrates lagrange polynomials to the points [nodes]
        M = self.M
        nodes_m, weights_m = self.gauss_legendre(np.ceil(M/2), b)  # use gauss-legendre quadrature to integrate polynomials
        weights = np.zeros(M)
        for j in np.arange(M):
            coeff = np.zeros(M)
            coeff[j] = 1.0  # is unity because it needs to be scaled with y_j for interpolation we have  sum y_j*l_j
            poly_coeffs = scipy.linalg.solve_triangular(self.NewtonVM(self.nodes), coeff, lower=True)
            eval_newt_poly = self.Horner_newton(poly_coeffs, self.nodes, nodes_m)
            weights[j] = np.dot(weights_m, eval_newt_poly)
        return weights

    def Qmatrix(self):
        """
        Integration Matrix
        """
        M = self.M
        self.Q = np.zeros([M, M])

        # for all nodes, get weights for the interval [tleft,node]
        for m in np.arange(M):
            w = self.get_weights(self.nodes[m])
            self.Q[m, 0:] = w

    def Smatrix(self):
        """
        Integration matrix based on Q: sum(S@vector) returns integration
        """
        from copy import deepcopy
        M = self.M
        self.S = np.zeros([M, M])

        self.S[0, :] = deepcopy(self.Q[0, :])
        for m in np.arange(1, M):
            self.S[m, :] = self.Q[m, :] - self.Q[m - 1, :]

    def matmul_UFL(self, a, b):
        # b is nx1 array!
        n = np.shape(a)[0]
        result = [float(0)]*n
        for j in range(n):
            for k in range(n):
                result[j] += float(a[j, k])*b[k]
            result[j] = assemble(result[j])
        return result

    @abstractmethod
    def apply(self, xin, xout):
        pass


class FE_SDC(SDC):

    def setup(self, equation, uadv=None):

        residual = equation.residual

        self.base = ForwardEuler(self.state)
        self.base.setup(equation, residual=residual)
        self.residual = self.base.residual

        # set up SDC form and solver
        W = equation.function_space
        dt = self.dt
        self.W = W
        self.Unodes = [Function(W) for _ in range(self.M+1)]
        self.Unodes1 = [Function(W) for _ in range(self.M+1)]
        self.fUnodes = [Function(W) for _ in range(self.M+1)]

        self.U_SDC = Function(W)
        self.U0 = Function(W)
        self.Un = Function(W)
        self.Q_ = Function(W)

        F = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: dt*t)

        a = F.label_map(lambda t: t.has_label(time_derivative),
                        replace_subject(self.U_SDC),
                        drop)

        F_exp = F.label_map(all_terms, replace_subject(self.Un))
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)

        F0 = F.label_map(lambda t: t.has_label(time_derivative),
                         drop,
                         replace_subject(self.U0))
        F0 = F0.label_map(all_terms,
                          lambda t: -1*t)

        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_),
                                    drop)

        F_SDC = a + F_exp + F0 + Q

        bcs = [DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain) for bc in equation.bcs['u']]
        prob_SDC = NonlinearVariationalProblem(F_SDC.form, self.U_SDC, bcs=bcs)
        self.solver_SDC = NonlinearVariationalSolver(prob_SDC)

        # set up RHS evaluation
        self.Urhs = Function(W)
        self.Uin = Function(W)
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs),
                                    drop)
        L = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    drop,
                                    replace_subject(self.Uin))
        Frhs = a - L
        prob_rhs = NonlinearVariationalProblem(Frhs.form, self.Urhs, bcs=bcs)
        self.solver_rhs = NonlinearVariationalSolver(prob_rhs)

    def apply(self, xin, xout):
        self.Un.assign(xin)

        self.Unodes[0].assign(self.Un)
        for m in range(self.M):
            self.base.dt.assign(self.dtau[m])
            self.base.apply(self.Unodes[m], self.Unodes[m+1])

        k = 0
        while k < self.maxk:
            k += 1

            for m in range(1, self.M+1):
                self.Uin.assign(self.Unodes[m])
                self.solver_rhs.solve()
                self.fUnodes[m-1].assign(self.Urhs)

            quad = self.matmul_UFL(self.S, self.fUnodes)
            # quad = dot(as_matrix(S),
            #            as_vector([f(Unodes[1]), f(Unodes[2]), f(Unodes[3])]))

            self.Unodes1[0].assign(self.Unodes[0])
            for m in range(1, self.M+1):
                self.dt.assign(self.dtau[m-1])
                self.U0.assign(self.Unodes[m-1])
                self.Un.assign(self.Unodes1[m-1])
                self.Q_.assign(quad[m-1])
                self.solver_SDC.solve()
                self.Unodes1[m].assign(self.U_SDC)
            for m in range(1, self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])

            self.Un.assign(self.Unodes1[-1])
            # print(k, self.Un.split()[1].dat.data.max())
        if self.maxk > 0:
            xout.assign(self.Un)
        else:
            xout.assign(self.Unodes[-1])


class BE_SDC(SDC):

    def setup(self, equation, uadv=None):

        residual = equation.residual

        self.base = BackwardEuler(self.state)

        uadv = self.state.fields("u")

        self.base.setup(equation, uadv=uadv, residual=residual)
        self.residual = self.base.residual

        # set up SDC form and solver
        W = equation.function_space
        dt = self.dt
        self.W = W
        self.Unodes = [Function(W) for _ in range(self.M+1)]
        self.Unodes1 = [Function(W) for _ in range(self.M+1)]
        self.fUnodes = [Function(W) for _ in range(self.M+1)]

        self.U_SDC = Function(W)
        self.U0 = Function(W)
        self.U01 = Function(W)
        self.Un = Function(W)
        self.Q_ = Function(W)

        F = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: dt*t)

        F_imp = F.label_map(all_terms,
                            replace_subject(self.U_SDC))

        F_exp = F.label_map(lambda t: t.has_label(time_derivative),
                            replace_subject(self.Un),
                            drop)
        F_exp = F_exp.label_map(all_terms,
                                lambda t: -1*t)

        F01 = F.label_map(lambda t: t.has_label(time_derivative),
                          drop,
                          replace_subject(self.U01))

        F01 = F01.label_map(all_terms, lambda t: -1*t)

        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_),
                                    drop)

        F_SDC = F_imp + F_exp + F01 + Q

        try:
            bcs = equation.bcs['u']
        except KeyError:
            bcs = None
        prob_SDC = NonlinearVariationalProblem(F_SDC.form, self.U_SDC, bcs=bcs)
        self.solver_SDC = NonlinearVariationalSolver(prob_SDC)

        # set up RHS evaluation
        self.Urhs = Function(W)
        self.Uin = Function(W)
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs),
                                    drop)
        L = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    drop,
                                    replace_subject(self.Uin))
        Frhs = a - L
        prob_rhs = NonlinearVariationalProblem(Frhs.form, self.Urhs, bcs=bcs)
        self.solver_rhs = NonlinearVariationalSolver(prob_rhs)

    def apply(self, xin, xout):
        self.Un.assign(xin)

        self.Unodes[0].assign(self.Un)
        for m in range(self.M):
            self.base.dt.assign(self.dtau[m])
            self.base.apply(self.Unodes[m], self.Unodes[m+1])

        k = 0
        while k < self.maxk:
            k += 1

            self.fUnodes = []
            for m in range(1, self.M+1):
                self.Uin.assign(self.Unodes[m])
                self.solver_rhs.solve()
                self.fUnodes[m-1].assign(self.Urhs)

            quad = self.matmul_UFL(self.S, self.fUnodes)
            # quad = dot(as_matrix(S),
            #            as_vector([f(Unodes[1]), f(Unodes[2]), f(Unodes[3])]))

            self.Unodes1[0].assign(self.Unodes[0])
            for m in range(1, self.M+1):
                self.dt.assign(self.dtau[m-1])
                self.U0.assign(self.Unodes[m-1])
                self.U01.assign(self.Unodes[m])
                self.Un.assign(self.Unodes1[m-1])
                self.Q_.assign(quad[m-1])
                self.solver_SDC.solve()
                self.Unodes1[m].assign(self.U_SDC)
            for m in range(1, self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])

            self.Un.assign(self.Unodes1[-1])
            #print(k, self.Un.split()[1].dat.data.max())
        if self.maxk > 0:
            xout.assign(self.Un)
        else:
            xout.assign(self.Unodes[-1])


class IMEX_SDC(SDC):

    def setup(self, equation, uadv=None):

        residual = equation.residual

        residual = residual.label_map(
            lambda t: any(t.has_label(time_derivative, advection)),
            map_if_false=lambda t: implicit(t))

        residual = residual.label_map(
            lambda t: t.has_label(advection),
            lambda t: explicit(t))

        self.IMEX = IMEX_Euler(self.state)
        self.IMEX.setup(equation, residual=residual)
        self.residual = self.IMEX.residual

        # set up SDC form and solver
        W = equation.function_space
        dt = self.dt
        self.W = W
        self.Unodes = [Function(W) for _ in range(self.M+1)]
        self.Unodes1 = [Function(W) for _ in range(self.M+1)]
        self.fUnodes = [Function(W) for _ in range(self.M+1)]

        self.U_SDC = Function(W)
        self.U0 = Function(W)
        self.U01 = Function(W)
        self.Un = Function(W)
        self.Q_ = Function(W)

        F = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    map_if_false=lambda t: dt*t)

        F_imp = F.label_map(lambda t: any(t.has_label(time_derivative, implicit)),
                            replace_subject(self.U_SDC),
                            drop)

        F_exp = F.label_map(lambda t: any(t.has_label(time_derivative, explicit)),
                            replace_subject(self.Un),
                            drop)
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)

        F01 = F.label_map(lambda t: t.has_label(implicit),
                          replace_subject(self.U01),
                          drop)

        F01 = F01.label_map(all_terms, lambda t: -1*t)

        F0 = F.label_map(lambda t: t.has_label(explicit),
                         replace_subject(self.U0),
                         drop)
        F0 = F0.label_map(all_terms, lambda t: -1*t)

        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_),
                                    drop)

        F_SDC = F_imp + F_exp + F01 + F0 + Q

        bcs = equation.bcs['u']
        prob_SDC = NonlinearVariationalProblem(F_SDC.form, self.U_SDC, bcs=bcs)
        self.solver_SDC = NonlinearVariationalSolver(prob_SDC)

        # set up RHS evaluation
        self.Urhs = Function(W)
        self.Uin = Function(W)
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs),
                                    drop)
        L = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    drop,
                                    replace_subject(self.Uin.split()))
        Frhs = a - L
        prob_rhs = NonlinearVariationalProblem(Frhs.form, self.Urhs, bcs=bcs)
        self.solver_rhs = NonlinearVariationalSolver(prob_rhs)

    def apply(self, xin, xout):
        self.Un.assign(xin)

        self.Unodes[0].assign(self.Un)
        for m in range(self.M):
            self.IMEX.dt.assign(self.dtau[m])
            self.IMEX.apply(self.Unodes[m], self.Unodes[m+1])

        k = 0
        while k < self.maxk:
            k += 1

            for m in range(1, self.M+1):
                self.Uin.assign(self.Unodes[m])
                self.solver_rhs.solve()
                self.fUnodes[m-1].assign(self.Urhs)

            quad = self.matmul_UFL(self.S, self.fUnodes)
            # quad = dot(as_matrix(S),
            #            as_vector([f(Unodes[1]), f(Unodes[2]), f(Unodes[3])]))

            self.Unodes1[0].assign(self.Unodes[0])
            for m in range(1, self.M+1):
                self.dt.assign(self.dtau[m-1])
                self.U0.assign(self.Unodes[m-1])
                self.U01.assign(self.Unodes[m])
                self.Un.assign(self.Unodes1[m-1])
                self.Q_.assign(quad[m-1])
                self.solver_SDC.solve()
                self.Unodes1[m].assign(self.U_SDC)
            for m in range(1, self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])

            self.Un.assign(self.Unodes1[-1])
            print(k, self.Un.split()[1].dat.data.max())
        if self.maxk > 0:
            xout.assign(self.Un)
        else:
            xout.assign(self.Unodes[-1])
