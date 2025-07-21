"""
Objects for discretising time derivatives using Deferred Correction (DC)
Methods. This includes Spectral Deferred Correction (SDC) and Serial Revisionist
Integral Deferred Correction (RIDC) methods.

These methods discretise ∂y/∂t = F(y), for variable y, time t, and operator F.

In Picard integral form, this equation is:
y(t) = y_n + ∫[t_n, t] F(y(s)) ds

================================================================================
Spectral Deferred Correction (SDC) Formulation:
================================================================================

SDC methods integrate the function F(y) over the interval [t_n, t_n+1] using
quadrature. Evaluating y on temporal quadrature nodes gives:
y_m = y_n + Σ[j=1,M] q_mj * F(y_j)
where q_mj are derived from integrating Lagrange polynomials, similar to how
Runge-Kutta methods are constructed.

In matrix form:
(I - dt * Q * F)(y) = y_n

Using Picard iteration:
y^(k+1) = y^k + (y_n - (I - dt * Q * F)(y^k))

Preconditioning this system with an approximation Q_delta gives:
(I - dt * Q_delta * F)(y^(k+1)) = y_n + dt * (Q - Q_delta) * F(y^k)

Two formulations are commonly used:
1. Zero-to-node (Z2N):
    y_m^(k+1) = y_n + Σ[j=1,M] q'_mj * (F(y_j^(k+1)) - F(y_j^k))
                    + Σ[j=1,M] q_mj * F(y_(j)^k)
    where q_mj are entries in Q and q'_mj are entries in Q_delta.

2. Node-to-node (N2N):
    y_m^(k+1) = y_(m-1)^(k+1) + dtau_m * (F(y_(m)^(k+1)) - F(y_(m)^k))
                    + Σ[j=1,M] s_mj * F(y_(j)^k)
    where s_mj = q_mj - q_(m-1)j for entries q_ik in Q.

Key choices in SDC:
- Quadrature node type (e.g., Gauss-Lobatto)
- Number of quadrature nodes
- Number of iterations (each iteration increases accuracy up to the quadrature order)
- Choice of Q_delta (e.g., Forward Euler, Backward Euler, LU-trick)
- Initial solution on quadrature nodes

================================================================================
Revisionist Integral Deferred Correction (RIDC) Formulation:
================================================================================

RIDC methods are similar to SDC but use equidistant nodes and a different
formulation for the error equation. The process involves:
1. Using a low-order method (predictor) to compute an initial solution:
    y_m^(0) = y_(m-1)^(0) + dt * F(y_(m)^(0))

2. Performing K correction steps:
    y_m^(k+1) = y_(m-1)^(k+1) + dt * (F(y_(m)^(k+1)) - F(y_(m)^k))
                    + Σ[j=1,M] s_mj * F(y_(j)^k)
We solve on N equispaced nodes on the interval [0, T] divided into J intervals,
each further divided into M subintervals:

     0 * * * * * | * * * * * | * * * * * | * * * * * | * * * * * T
     |   J intervals, each with M subintervals                   |

Here, M >> K, and M must be at least K * (K+1) / 2 for the reduced stencil RIDC method.
dt = T / N, N = J * M.
Each correction sweep increases accuracy up to the quadrature order.

Key choices in RIDC:
- Number of subintervals J
- Number of quadrature nodes M + 1
- Number of correction iterations K
"""

from abc import ABCMeta
import numpy as np
from firedrake import (
    Function, NonlinearVariationalProblem, NonlinearVariationalSolver, Constant
)
from firedrake.fml import (
    replace_subject, all_terms, drop
)
from firedrake.utils import cached_property
from gusto.time_discretisation.time_discretisation import wrapper_apply
from gusto.core.labels import (time_derivative, implicit, explicit, source_label)
from qmat import genQCoeffs, genQDeltaCoeffs

__all__ = ["SDC", "RIDC"]


class SDC(object, metaclass=ABCMeta):
    """Class for Spectral Deferred Correction schemes."""

    def __init__(self, base_scheme, domain, M, maxk, quad_type, node_type, qdelta_imp, qdelta_exp,
                 formulation="N2N", field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None, final_update=True,
                 limiter=None, options=None, initial_guess="base"):
        """
        Initialise SDC object
        Args:
            base_scheme (:class:`TimeDiscretisation`): Base time stepping scheme to get first guess of solution on
                quadrature nodes.
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            M (int): Number of quadrature nodes to compute spectral integration over
            maxk (int): Max number of correction interations
            quad_type (str): Type of quadrature to be used. Options are
                GAUSS, RADAU-LEFT, RADAU-RIGHT and LOBATTO
            node_type (str): Node type to be used. Options are
                EQUID, LEGENDRE, CHEBY-1, CHEBY-2, CHEBY-3 and CHEBY-4
            qdelta_imp (str): Implicit Qdelta matrix to be used. Options are
                BE, LU, TRAP, EXACT, PIC, OPT, WEIRD, MIN-SR-NS, MIN-SR-S
            qdelta_exp (str): Explicit Qdelta matrix to be used. Options are
                FE, EXACT, PIC
            formulation (str, optional): Whether to use node-to-node or zero-to-node
                formulation. Options are N2N and Z2N. Defaults to N2N
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            linear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying linear solver. Defaults to None.
            nonlinear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying nonlinear solver. Defaults to None.
            final_update (bool, optional): Whether to compute final update, or just take last
                quadrature value. Defaults to True
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
                options to either be passed to the spatial discretisation, or
                to control the "wrapper" methods, such as Embedded DG or a
                recovery method. Defaults to None.
            initial_guess (str, optional): Initial guess to be base timestepper, or copy
        """
        # Check the configuration options
        if (not (formulation == "N2N" or formulation == "Z2N")):
            raise ValueError('Formulation not implemented')

        # Initialise parameters
        self.base = base_scheme
        self.base.dt = domain.dt
        self.field_name = field_name
        self.domain = domain
        self.dt_coarse = domain.dt
        self.M = M
        self.maxk = maxk
        self.final_update = final_update
        self.formulation = formulation
        self.limiter = limiter
        self.augmentation = self.base.augmentation
        self.wrapper = self.base.wrapper

        # Get quadrature nodes and weights
        self.nodes, self.weights, self.Q = genQCoeffs("Collocation", nNodes=M,
                                                      nodeType=node_type,
                                                      quadType=quad_type,
                                                      form=formulation)

        # Rescale to be over [0,dt] rather than [0,1]
        self.nodes = float(self.dt_coarse)*self.nodes

        self.dtau = np.diff(np.append(0, self.nodes))
        self.Q = float(self.dt_coarse)*self.Q
        self.Qfin = float(self.dt_coarse)*self.weights
        self.qdelta_imp_type = qdelta_imp
        self.formulation = formulation
        self.node_type = node_type
        self.quad_type = quad_type

        # Get Q_delta matrices
        self.Qdelta_imp = genQDeltaCoeffs(qdelta_imp, form=formulation,
                                          nodes=self.nodes, Q=self.Q, nNodes=M, nodeType=node_type, quadType=quad_type)
        self.Qdelta_exp = genQDeltaCoeffs(qdelta_exp, form=formulation,
                                          nodes=self.nodes, Q=self.Q, nNodes=M, nodeType=node_type, quadType=quad_type)
        # Set default linear and nonlinear solver options if none passed in
        if linear_solver_parameters is None:
            self.linear_solver_parameters = {'snes_type': 'ksponly',
                                             'ksp_type': 'cg',
                                             'pc_type': 'bjacobi',
                                             'sub_pc_type': 'ilu'}
        else:
            self.linear_solver_parameters = linear_solver_parameters

        if nonlinear_solver_parameters is None:
            self.nonlinear_solver_parameters = {'snes_type': 'newtonls',
                                                'ksp_type': 'gmres',
                                                'pc_type': 'bjacobi',
                                                'sub_pc_type': 'ilu'}
        else:
            self.nonlinear_solver_parameters = nonlinear_solver_parameters

        # Flag to check wheter initial guess is generated using base time discretisation
        # (i.e. Forward Euler)
        if (initial_guess == "base"):
            self.base_flag = True
        else:
            self.base_flag = False

    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the SDC time discretisation based on the equation.n

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            apply_bcs (bool, optional): whether to apply the equation's boundary
                conditions. Defaults to True.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        # Inherit from base time discretisation
        self.base.setup(equation, apply_bcs, *active_labels)
        self.equation = self.base.equation
        self.residual = self.base.residual
        self.evaluate_source = self.base.evaluate_source

        for t in self.residual:
            # Check all terms are labeled implicit or explicit
            if ((not t.has_label(implicit)) and (not t.has_label(explicit))
               and (not t.has_label(time_derivative)) and (not t.has_label(source_label))):
                raise NotImplementedError("Non time-derivative or source terms must be labeled as implicit or explicit")

        # Set up bcs
        self.bcs = self.base.bcs

        # Set up SDC variables
        if self.field_name is not None and hasattr(equation, "field_names"):
            self.idx = equation.field_names.index(self.field_name)
            W = equation.spaces[self.idx]
        else:
            self.field_name = equation.field_name
            W = equation.function_space
            self.idx = None
        self.W = W
        self.Unodes = [Function(W) for _ in range(self.M+1)]
        self.Unodes1 = [Function(W) for _ in range(self.M+1)]
        self.fUnodes = [Function(W) for _ in range(self.M)]
        self.quad = [Function(W) for _ in range(self.M)]
        self.source_Uk = [Function(W) for _ in range(self.M+1)]
        self.source_Ukp1 = [Function(W) for _ in range(self.M+1)]
        self.U_DC = Function(W)
        self.U_start = Function(W)
        self.Un = Function(W)
        self.Q_ = Function(W)
        self.quad_final = Function(W)
        self.U_fin = Function(W)
        self.Urhs = Function(W)
        self.Uin = Function(W)
        self.source_in = Function(W)

    @property
    def nlevels(self):
        return 1

    def compute_quad(self):
        """
        Computes integration of F(y) on quadrature nodes
        """
        for j in range(self.M):
            self.quad[j].assign(0.)
            for k in range(self.M):
                self.quad[j] += float(self.Q[j, k])*self.fUnodes[k]

    def compute_quad_final(self):
        """
        Computes final integration of F(y) on quadrature nodes
        """
        self.quad_final.assign(0.)
        for k in range(self.M):
            self.quad_final += float(self.Qfin[k])*self.fUnodes[k]

    @property
    def res_rhs(self):
        """Set up the residual for the calculation of F(y)."""
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
        # F(y)
        L = self.residual.label_map(lambda t: any(t.has_label(time_derivative, source_label)),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        L_source = self.residual.label_map(lambda t: t.has_label(source_label),
                                           replace_subject(self.source_in, old_idx=self.idx),
                                           drop)
        residual_rhs = a - (L + L_source)
        return residual_rhs.form

    @property
    def res_fin(self):
        """Set up the residual for final solve."""
        # y_(n+1)
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.U_fin, old_idx=self.idx),
                                    drop)
        # y_n
        F_exp = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                        replace_subject(self.Un, old_idx=self.idx),
                                        drop)
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)

        # sum(j=1,M) q_j*F(y_j)
        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.quad_final, old_idx=self.idx),
                                    drop)

        residual_final = a + F_exp + Q
        return residual_final.form

    def res(self, m):
        """Set up the discretisation's residual for a given node m."""
        # Add time derivative terms  y^(k+1)_m - y_start for node m. y_start is y_n for Z2N formulation
        # and y^(k)_m for N2N formulation
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)
        residual = mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.U_DC, old_idx=self.idx))
        residual -= mass_form.label_map(all_terms,
                                        map_if_true=replace_subject(self.U_start, old_idx=self.idx))
        # Loop through nodes up to m-1 and calcualte
        # sum(j=1,m-1) Qdelta_imp[m,j]*(F(y_(m)^(k+1)) - F(y_(m)^k))
        for i in range(m):
            r_imp_kp1 = self.residual.label_map(
                lambda t: t.has_label(implicit),
                map_if_true=replace_subject(self.Unodes1[i+1], old_idx=self.idx),
                map_if_false=drop)
            r_imp_kp1 = r_imp_kp1.label_map(
                all_terms,
                lambda t: Constant(self.Qdelta_imp[m, i])*t)
            residual += r_imp_kp1
            r_imp_k = self.residual.label_map(
                lambda t: t.has_label(implicit),
                map_if_true=replace_subject(self.Unodes[i+1], old_idx=self.idx),
                map_if_false=drop)
            r_imp_k = r_imp_k.label_map(
                all_terms,
                lambda t: Constant(self.Qdelta_imp[m, i])*t)
            residual -= r_imp_k
        # Loop through nodes up to m-1 and calcualte
        #  sum(j=1,M)  Q_delta_exp[m,j]*(S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
        for i in range(self.M):
            r_exp_kp1 = self.residual.label_map(
                lambda t: t.has_label(explicit),
                map_if_true=replace_subject(self.Unodes1[i+1], old_idx=self.idx),
                map_if_false=drop)
            r_exp_kp1 = r_exp_kp1.label_map(
                all_terms,
                lambda t: Constant(self.Qdelta_exp[m, i])*t)

            residual += r_exp_kp1
            r_exp_k = self.residual.label_map(
                lambda t: t.has_label(explicit),
                map_if_true=replace_subject(self.Unodes[i+1], old_idx=self.idx),
                map_if_false=drop)
            r_exp_k = r_exp_k.label_map(
                all_terms,
                lambda t: Constant(self.Qdelta_exp[m, i])*t)
            residual -= r_exp_k

            # Calculate source terms
            r_source_kp1 = self.residual.label_map(
                lambda t: t.has_label(source_label),
                map_if_true=replace_subject(self.source_Ukp1[i+1], old_idx=self.idx),
                map_if_false=drop)
            r_source_kp1 = r_source_kp1.label_map(
                all_terms,
                lambda t: Constant(self.Qdelta_exp[m, i])*t)
            residual += r_source_kp1

            r_source_k = self.residual.label_map(
                lambda t: t.has_label(source_label),
                map_if_true=replace_subject(self.source_Uk[i+1], old_idx=self.idx),
                map_if_false=drop)
            r_source_k = r_source_k.label_map(
                all_terms,
                map_if_true=lambda t: Constant(self.Qdelta_exp[m, i])*t)
            residual -= r_source_k

        # Add on final implicit terms
        # Qdelta_imp[m,m]*(F(y_(m)^(k+1)) - F(y_(m)^k))
        r_imp_kp1 = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.U_DC, old_idx=self.idx),
            map_if_false=drop)
        r_imp_kp1 = r_imp_kp1.label_map(
            all_terms,
            lambda t: Constant(self.Qdelta_imp[m, m])*t)
        residual += r_imp_kp1
        r_imp_k = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.Unodes[m+1], old_idx=self.idx),
            map_if_false=drop)
        r_imp_k = r_imp_k.label_map(
            all_terms,
            lambda t: Constant(self.Qdelta_imp[m, m])*t)
        residual -= r_imp_k

        # Add on error term. sum(j=1,M) q_mj*F(y_m^k) for Z2N formulation
        # and sum(j=1,M) s_mj*F(y_m^k) for N2N formulation, where s_mj = q_mj-q_m-1j
        # and s1j = q1j.
        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_, old_idx=self.idx),
                                    drop)
        residual += Q
        return residual.form

    @cached_property
    def solvers(self):
        """Set up a list of solvers for each problem at a node m."""
        solvers = []
        for m in range(self.M):
            # setup solver using residual defined in derived class
            problem = NonlinearVariationalProblem(self.res(m), self.U_DC, bcs=self.bcs)
            solver_name = self.field_name+self.__class__.__name__ + "%s" % (m)
            solvers.append(NonlinearVariationalSolver(problem, solver_parameters=self.nonlinear_solver_parameters, options_prefix=solver_name))
        return solvers

    @cached_property
    def solver_fin(self):
        """Set up the problem and the solver for final update."""
        # setup linear solver using final residual defined in derived class
        prob_fin = NonlinearVariationalProblem(self.res_fin, self.U_fin, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_final"
        return NonlinearVariationalSolver(prob_fin, solver_parameters=self.linear_solver_parameters,
                                          options_prefix=solver_name)

    @cached_property
    def solver_rhs(self):
        """Set up the problem and the solver for mass matrix inversion."""
        # setup linear solver using rhs residual defined in derived class
        prob_rhs = NonlinearVariationalProblem(self.res_rhs, self.Urhs, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_rhs"
        return NonlinearVariationalSolver(prob_rhs, solver_parameters=self.linear_solver_parameters,
                                          options_prefix=solver_name)

    @wrapper_apply
    def apply(self, x_out, x_in):
        self.Un.assign(x_in)
        self.U_start.assign(self.Un)
        solver_list = self.solvers

        # Compute initial guess on quadrature nodes with low-order
        # base timestepper
        self.Unodes[0].assign(self.Un)
        if (self.base_flag):
            for m in range(self.M):
                self.base.dt = float(self.dtau[m])
                self.base.apply(self.Unodes[m+1], self.Unodes[m])
        else:
            for m in range(self.M):
                self.Unodes[m+1].assign(self.Un)
        for m in range(self.M+1):
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[m], self.base.dt, x_out=self.source_Uk[m])

        # Iterate through correction sweeps
        k = 0
        while k < self.maxk:
            k += 1

            if self.qdelta_imp_type == "MIN-SR-FLEX":
                # Recompute Implicit Q_delta matrix for each iteration k
                self.Qdelta_imp = genQDeltaCoeffs(
                    self.qdelta_imp_type,
                    form=self.formulation,
                    nodes=self.nodes,
                    Q=self.Q,
                    nNodes=self.M,
                    nodeType=self.node_type,
                    quadType=self.quad_type,
                    k=k
                )

            # Compute for N2N: sum(j=1,M) (s_mj*F(y_m^k) +  s_mj*S(y_m^k))
            # for Z2N: sum(j=1,M) (q_mj*F(y_m^k) +  q_mj*S(y_m^k))
            for m in range(1, self.M+1):
                self.Uin.assign(self.Unodes[m])
                # Include source terms
                for evaluate in self.evaluate_source:
                    evaluate(self.Uin, self.base.dt, x_out=self.source_in)
                self.solver_rhs.solve()
                self.fUnodes[m-1].assign(self.Urhs)
            self.compute_quad()

            # Loop through quadrature nodes and solve
            self.Unodes1[0].assign(self.Unodes[0])
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[0], self.base.dt, x_out=self.source_Uk[0])
            for m in range(1, self.M+1):
                # Set Q or S matrix
                self.Q_.assign(self.quad[m-1])

                # Set initial guess for solver, and pick correct solver
                if (self.formulation == "N2N"):
                    self.U_start.assign(self.Unodes1[m-1])
                self.solver = solver_list[m-1]
                self.U_DC.assign(self.Unodes[m])

                # Compute
                # for N2N:
                # y_m^(k+1) = y_(m-1)^(k+1) + dtau_m*(F(y_(m)^(k+1)) - F(y_(m)^k)
                #             + S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
                #             + sum(j=1,M) s_mj*(F+S)(y^k)
                # for Z2N:
                # y_m^(k+1) = y^n + sum(j=1,m) Qdelta_imp[m,j]*(F(y_(m)^(k+1)) - F(y_(m)^k))
                #             + sum(j=1,M)  Q_delta_exp[m,j]*(S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
                self.solver.solve()
                self.Unodes1[m].assign(self.U_DC)

                # Evaluate source terms
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes1[m], self.base.dt, x_out=self.source_Ukp1[m])

                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m])
            for m in range(1, self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])
                self.source_Uk[m].assign(self.source_Ukp1[m])

        if self.maxk > 0:
            # Compute value at dt rather than final quadrature node tau_M
            if self.final_update:
                for m in range(1, self.M+1):
                    self.Uin.assign(self.Unodes1[m])
                    self.source_in.assign(self.source_Ukp1[m])
                    self.solver_rhs.solve()
                    self.fUnodes[m-1].assign(self.Urhs)
                self.compute_quad_final()
                # Compute y_(n+1) = y_n + sum(j=1,M) q_j*F(y_j)

                self.U_fin.assign(self.Unodes[-1])
                self.solver_fin.solve()
                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.U_fin)
                x_out.assign(self.U_fin)
            else:
                # Take value at final quadrature node dtau_M
                x_out.assign(self.Unodes[-1])
        else:
            x_out.assign(self.Unodes[-1])


class RIDC(object, metaclass=ABCMeta):
    """Class for Revisionist Integral Deferred Correction schemes."""

    def __init__(self, base_scheme, domain, M, K, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None, reduced=True):
        """
        Initialise RIDC object
        Args:
            base_scheme (:class:`TimeDiscretisation`): Base time stepping scheme to get first guess of solution on
                quadrature nodes.
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            M (int): Number of subintervals
            K (int): Max number of correction interations
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            linear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying linear solver. Defaults to None.
            nonlinear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying nonlinear solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            options (:class:`AdvectionOptions`, optional): an object containing
            reduced (bool, optional): whether to use reduced or full stencils for RIDC.
        """
        self.base = base_scheme
        self.field_name = field_name
        self.domain = domain
        self.dt_coarse = domain.dt
        self.limiter = limiter
        self.augmentation = self.base.augmentation
        self.wrapper = self.base.wrapper
        self.K = K
        self.M = M
        self.reduced = reduced
        self.dt = Constant(float(self.dt_coarse)/(self.M))

        if reduced:
            self.Q = []
            for l in range(1, self.K+1):
                _, _, Q = genQCoeffs("Collocation", nNodes=l+1,
                                                      nodeType="EQUID",
                                                      quadType="LOBATTO",
                                                      form="N2N")
                Q = l* float(self.dt) * Q
                self.Q.append(Q)
        else:
            # Get integration weights
            _, _, self.Q = genQCoeffs("Collocation", nNodes=K+1,
                                                      nodeType="EQUID",
                                                      quadType="LOBATTO",
                                                      form="N2N")
            self.Q = self.K*float(self.dt)*self.Q

        # Set default linear and nonlinear solver options if none passed in
        if linear_solver_parameters is None:
            self.linear_solver_parameters = {'snes_type': 'ksponly',
                                             'ksp_type': 'cg',
                                             'pc_type': 'bjacobi',
                                             'sub_pc_type': 'ilu'}
        else:
            self.linear_solver_parameters = linear_solver_parameters

        if nonlinear_solver_parameters is None:
            self.nonlinear_solver_parameters = {'snes_type': 'newtonls',
                                                'ksp_type': 'gmres',
                                                'pc_type': 'bjacobi',
                                                'sub_pc_type': 'ilu'}
        else:
            self.nonlinear_solver_parameters = nonlinear_solver_parameters

    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the RIDC time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            apply_bcs (bool, optional): whether to apply the equation's boundary
                conditions. Defaults to True.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation to include.
        """
        # Inherit from base time discretisation
        self.base.setup(equation, apply_bcs, *active_labels)
        self.equation = self.base.equation
        self.residual = self.base.residual
        self.evaluate_source = self.base.evaluate_source

        for t in self.residual:
            # Check all terms are labeled implicit or explicit
            if ((not t.has_label(implicit)) and (not t.has_label(explicit))
               and (not t.has_label(time_derivative)) and (not t.has_label(source_label))):
                raise NotImplementedError("Non time-derivative or source terms must be labeled as implicit or explicit")

        # Set up bcs
        self.bcs = self.base.bcs

        # Set up RIDC variables
        if self.field_name is not None and hasattr(equation, "field_names"):
            self.idx = equation.field_names.index(self.field_name)
            W = equation.spaces[self.idx]
        else:
            self.field_name = equation.field_name
            W = equation.function_space
            self.idx = None
        self.W = W
        self.Unodes = [Function(W) for _ in range(self.M+1)]
        self.Unodes1 = [Function(W) for _ in range(self.M+1)]
        self.fUnodes = [Function(W) for _ in range(self.M+1)]
        self.quad = [Function(W) for _ in range(self.M+1)]
        self.source_Uk = [Function(W) for _ in range(self.M+1)]
        self.source_Ukp1 = [Function(W) for _ in range(self.M+1)]
        self.U_DC = Function(W)
        self.U_start = Function(W)
        self.Un = Function(W)
        self.Q_ = Function(W)
        self.quad_final = Function(W)
        self.U_fin = Function(W)
        self.Urhs = Function(W)
        self.Uin = Function(W)
        self.source_in = Function(W)
        self.source_Ukp1_m = Function(W)
        self.source_Uk_m = Function(W)
        self.Uk_mp1 = Function(W)
        self.Uk_m = Function(W)
        self.Ukp1_m = Function(W)

    @property
    def nlevels(self):
        return 1

    def equidistant_nodes(self, M):
        """
        Returns a grid of M equispaced nodes from -1 to 1
        """
        grid = np.linspace(-1., 1., M)
        return grid

    def lagrange_polynomial(self, index, nodes):
        """
        Returns the coefficients of the Lagrange polynomial l_m with m=index
        """

        M = len(nodes)

        # c is the denominator
        c = 1.
        for k in range(M):
            if k != index:
                c *= (nodes[index] - nodes[k])

        coeffs = np.zeros(M)
        coeffs[0] = 1.
        m = 0

        for k in range(M):
            if k != index:
                m += 1
                d1 = np.zeros(M)
                d2 = np.zeros(M)

                d1 = (-1.)*nodes[k] * coeffs
                d2[1:m+1] = coeffs[0:m]

                coeffs = d1+d2
        return coeffs / c

    def integrate_polynomial(self, p):
        """
        Given a list of coefficients of a polynomial p,
        this returns those of the integral of p
        """
        integral_coeffs = np.zeros(len(p)+1)

        for n, pn in enumerate(p):
            integral_coeffs[n+1] = 1/(n+1) * pn

        return integral_coeffs

    def evaluate(self, p, a, b):
        """
        Given a list of coefficients of a polynomial p, this returns the value of p(b)-p(a)
        """
        value = 0.
        for n, pn in enumerate(p):
            value += pn * (b**n - a**n)

        return value

    def lagrange_integration_matrix(self, M):
        """
        Returns the integration matrix for the Lagrange polynomial of order M
        """

        # Set up equidistant nodes and initialise matrix to zero
        nodes = self.equidistant_nodes(M)
        L = len(nodes)
        int_matrix = np.zeros((L, L))

        # Fill in matrix values
        for index in range(L):
            coeff_p = self.lagrange_polynomial(index, nodes)
            int_coeff = self.integrate_polynomial(coeff_p)

            for n in range(L-1):
                int_matrix[n+1, index] = self.evaluate(int_coeff, nodes[n], nodes[n+1])

        return int_matrix

    def compute_quad(self, Q, fUnodes, m):
        """
        Computes integration of F(y) on quadrature nodes
        """
        quad = Function(self.W)
        quad.assign(0.)
        for k in range(0, np.shape(Q)[1]):
            quad += float(Q[m, k])*fUnodes[k]
        return quad

    def compute_quad_final(self, Q, fUnodes, m):
        """
        Computes final integration of F(y) on quadrature nodes
        """
        quad = Function(self.W)
        quad.assign(0.)
        if self.reduced:
            l = np.shape(Q)[0] - 1
        else:
            l = self.K
        for k in range(0, l+1):
            quad += float(Q[-1, k])*fUnodes[m - l + k]
        return quad

    @property
    def res_rhs(self):
        """Set up the residual for the calculation of F(y)."""
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
        # F(y)
        L = self.residual.label_map(lambda t: any(t.has_label(time_derivative, source_label)),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        L_source = self.residual.label_map(lambda t: t.has_label(source_label),
                                           replace_subject(self.source_in, old_idx=self.idx),
                                           drop)
        residual_rhs = a - (L + L_source)
        return residual_rhs.form

    @property
    def res(self):
        """Set up the discretisation's residual."""
        # Add time derivative terms  y^(k+1)_m - y_n
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)
        residual = mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.U_DC, old_idx=self.idx))
        residual -= mass_form.label_map(all_terms,
                                        map_if_true=replace_subject(self.U_start, old_idx=self.idx))

        # Calculate source terms
        r_source_kp1 = self.residual.label_map(
            lambda t: t.has_label(source_label),
            map_if_true=replace_subject(self.source_Ukp1_m, old_idx=self.idx),
            map_if_false=drop)
        r_source_kp1 = r_source_kp1.label_map(
            all_terms,
            lambda t: Constant(self.dt)*t)
        residual += r_source_kp1

        r_source_k = self.residual.label_map(
            lambda t: t.has_label(source_label),
            map_if_true=replace_subject(self.source_Uk_m, old_idx=self.idx),
            map_if_false=drop)
        r_source_k = r_source_k.label_map(
            all_terms,
            map_if_true=lambda t: Constant(self.dt)*t)
        residual -= r_source_k

        # Add on final implicit terms
        # dt*(F(y_(m)^(k+1)) - F(y_(m)^k))
        r_imp_kp1 = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.U_DC, old_idx=self.idx),
            map_if_false=drop)
        r_imp_kp1 = r_imp_kp1.label_map(
            all_terms,
            lambda t: Constant(self.dt)*t)
        residual += r_imp_kp1
        r_imp_k = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.Uk_mp1, old_idx=self.idx),
            map_if_false=drop)
        r_imp_k = r_imp_k.label_map(
            all_terms,
            lambda t: Constant(self.dt)*t)
        residual -= r_imp_k

        r_exp_kp1 = self.residual.label_map(
            lambda t: t.has_label(explicit),
            map_if_true=replace_subject(self.Ukp1_m, old_idx=self.idx),
            map_if_false=drop)
        r_exp_kp1 = r_exp_kp1.label_map(
            all_terms,
            lambda t: Constant(self.dt)*t)
        residual += r_exp_kp1
        r_exp_k = self.residual.label_map(
            lambda t: t.has_label(explicit),
            map_if_true=replace_subject(self.Uk_m, old_idx=self.idx),
            map_if_false=drop)
        r_exp_k = r_exp_k.label_map(
            all_terms,
            lambda t: Constant(self.dt)*t)
        residual -= r_exp_k

        # Add on sum(j=1,M) s_mj*F(y_m^k), where s_mj = q_mj-q_m-1j
        # and s1j = q1j.
        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_, old_idx=self.idx),
                                    drop)
        residual += Q
        return residual.form

    @cached_property
    def solver(self):
        """Set up the problem and the solver for the nonlinear solve."""
        # setup solver using residual defined in derived class
        problem = NonlinearVariationalProblem(self.res, self.U_DC, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__
        solver = NonlinearVariationalSolver(problem, solver_parameters=self.nonlinear_solver_parameters, options_prefix=solver_name)
        return solver

    @cached_property
    def solver_rhs(self):
        """Set up the problem and the solver for mass matrix inversion."""
        # setup linear solver using rhs residual defined in derived class
        prob_rhs = NonlinearVariationalProblem(self.res_rhs, self.Urhs, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_rhs"
        return NonlinearVariationalSolver(prob_rhs, solver_parameters=self.linear_solver_parameters,
                                          options_prefix=solver_name)

    @wrapper_apply
    def apply(self, x_out, x_in):
        self.Un.assign(x_in)

        # Compute initial guess on quadrature nodes with low-order
        # base timestepper
        self.Unodes[0].assign(self.Un)
        self.M1 = self.K

        for m in range(self.M):
            self.base.dt = float(self.dt)
            self.base.apply(self.Unodes[m+1], self.Unodes[m])

        for m in range(self.M+1):
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[m], self.base.dt, x_out=self.source_Uk[m])

        # Iterate through correction sweeps
        for k in range(1, self.K+1):
            # Compute: sum(j=1,M) (s_mj*F(y_m^k) +  s_mj*S(y_m^k))
            for m in range(self.M+1):
                self.Uin.assign(self.Unodes[m])
                # Include source terms
                for evaluate in self.evaluate_source:
                    evaluate(self.Uin, self.base.dt, x_out=self.source_in)
                self.solver_rhs.solve()
                self.fUnodes[m].assign(self.Urhs)

            # Loop through quadrature nodes and solve
            self.Unodes1[0].assign(self.Unodes[0])
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[0], self.base.dt, x_out=self.source_Uk[0])
            if self.reduced:
                self.M1 = k
            for m in range(0, self.M1):
                # Set integration matrix
                if self.reduced:
                    self.Q_.assign(self.compute_quad(self.Q[k-1], self.fUnodes, m+1))
                else:
                    self.Q_.assign(self.compute_quad(self.Q, self.fUnodes, m+1))

                # Set initial guess for solver, and pick correct solver
                self.U_start.assign(self.Unodes1[m])
                self.Ukp1_m.assign(self.Unodes1[m])
                self.Uk_mp1.assign(self.Unodes[m+1])
                self.Uk_m.assign(self.Unodes[m])
                self.source_Ukp1_m.assign(self.source_Ukp1[m])
                self.source_Uk_m.assign(self.source_Uk[m])
                self.U_DC.assign(self.Unodes[m+1])

                # Compute:
                # y_m^(k+1) = y_(m-1)^(k+1) + dt*(F(y_(m)^(k+1)) - F(y_(m)^k)
                #             + S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
                #             + sum(j=1,M) s_mj*(F+S)(y^k)
                self.solver.solve()
                self.Unodes1[m+1].assign(self.U_DC)

                # Evaluate source terms
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes1[m+1], self.base.dt, x_out=self.source_Ukp1[m+1])

                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m+1])
            for m in range(self.M1, self.M):
                # Set integration matrix
                if self.reduced:
                    self.Q_.assign(self.compute_quad_final(self.Q[k-1], self.fUnodes, m+1))
                else:
                    self.Q_.assign(self.compute_quad_final(self.Q, self.fUnodes, m+1))

                # Set initial guess for solver, and pick correct solver
                self.U_start.assign(self.Unodes1[m])
                self.Ukp1_m.assign(self.Unodes1[m])
                self.Uk_mp1.assign(self.Unodes[m+1])
                self.Uk_m.assign(self.Unodes[m])
                self.source_Ukp1_m.assign(self.source_Ukp1[m])
                self.source_Uk_m.assign(self.source_Uk[m])
                self.U_DC.assign(self.Unodes[m+1])

                # Compute:
                # y_m^(k+1) = y_(m-1)^(k+1) + dt*(F(y_(m)^(k+1)) - F(y_(m)^k)
                #             + S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
                #             + sum(j=1,M) s_mj*(F+S)(y^k)
                self.solver.solve()
                self.Unodes1[m+1].assign(self.U_DC)

                # Evaluate source terms
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes1[m+1], self.base.dt, x_out=self.source_Ukp1[m+1])

                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m+1])

            for m in range(self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])
                self.source_Uk[m].assign(self.source_Ukp1[m])

        x_out.assign(self.Unodes[-1])
