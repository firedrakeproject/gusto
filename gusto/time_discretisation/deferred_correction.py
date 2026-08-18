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
from functools import cached_property
import numpy as np
from firedrake import (
    Function, Cofunction, NonlinearVariationalProblem, NonlinearVariationalSolver,
    Constant, derivative
)
from firedrake.assemble import get_assembler
from firedrake.fml import (
    replace_subject, all_terms, drop
)
from gusto.time_discretisation.time_discretisation import wrapper_apply
from gusto.core.labels import (time_derivative, implicit, explicit, source_label)
from qmat import genQCoeffs, genQDeltaCoeffs
from gusto.solvers.solver_presets import hybridised_solver_parameters
from qmat.qdelta.diag import MIN_SR_FLEX
from gusto.core.logging import logger

__all__ = ["SDC", "RIDC"]


class SDC(object, metaclass=ABCMeta):
    """Class for Spectral Deferred Correction schemes."""

    def __init__(self, base_scheme, domain, M, maxk, quad_type, node_type, qdelta_imp, qdelta_exp,
                 formulation="N2N", field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None, final_update=True,
                 limiter=None, initial_guess="base", sweep_tols=None):
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
                BE, LU, TRAP, EXACT, PIC, OPT, WEIRD, MIN-SR-NS, MIN-SR-S, MIN-SR-FLEX
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
            initial_guess (str, optional): Initial guess to be base timestepper, or copy
            sweep_tols (list of dict, optional): list of tolerances for each sweep. Defaults to None.
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

        self.qdelta_imp_type = qdelta_imp
        self.formulation = formulation
        self.node_type = node_type
        self.quad_type = quad_type

        # Get Q_delta matrices
        self.Qdelta_imp = genQDeltaCoeffs(qdelta_imp, form=formulation,
                                          nodes=self.nodes, Q=self.Q, nNodes=M, nodeType=node_type, quadType=quad_type, k=1)
        self.Qdelta_exp = genQDeltaCoeffs(qdelta_exp, form=formulation,
                                          nodes=self.nodes, Q=self.Q, nNodes=M, nodeType=node_type, quadType=quad_type)

        # Rescale to be over [0,dt] rather than [0,1]
        self.nodes = float(self.dt_coarse)*self.nodes
        self.dtau = np.diff(np.append(0, self.nodes))
        self.Q = float(self.dt_coarse)*self.Q
        self.Qfin = float(self.dt_coarse)*self.weights
        self.Qdelta_imp = float(self.dt_coarse)*self.Qdelta_imp
        self.Qdelta_exp = float(self.dt_coarse)*self.Qdelta_exp

        # If both Qdelta matrices are purely diagonal (lower triangle all-zero), each node's
        # correction is independent. On the final sweep with no final update, only the last
        # node needs solving since intermediate nodes don't contribute to the output.
        _lower_imp = np.tril(self.Qdelta_imp[:self.M, :self.M], k=-1)
        _lower_exp = np.tril(self.Qdelta_exp[:self.M, :self.M], k=-1)
        self._final_sweep_shortcut = (np.allclose(_lower_imp, 0) and np.allclose(_lower_exp, 0))

        # Set default linear and nonlinear solver options if none passed in
        if linear_solver_parameters is None:
            self.linear_solver_parameters = {'snes_type': 'ksponly',
                                             'ksp_type': 'cg',
                                             'pc_type': 'bjacobi',
                                             'sub_pc_type': 'ilu'}
        else:
            self.linear_solver_parameters = linear_solver_parameters

        self.nonlinear_solver_parameters = nonlinear_solver_parameters
        self.appctx = None



        # Flag to check wheter initial guess is generated using base time discretisation
        # (i.e. Forward Euler)
        if (initial_guess == "base"):
            self.base_flag = True
        else:
            self.base_flag = False

        # Set up counters for total iterations and get sweep tolerances
        self.total_ksp_its = 0
        self.total_snes_its = 0
        self.sweep_tols = sweep_tols
        self._step_count = 1

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
        self.source_Uk = [Function(W) for _ in range(self.M+1)]
        self.source_Ukp1 = [Function(W) for _ in range(self.M+1)]
        self.U_DC = Function(W)
        self.U_start = Function(W)
        self.Un = Function(W)
        self.U_fin = Function(W)

        # Qf[m] = sum_i Q[m,i] * F(Unodes[i+1]), where F is the
        # non-time-derivative part of the residual. Assembled directly
        # from a single per-node UFL form  into
        # a Cofunction each sweep. Qf_fin is the equivanlent for the final update.
        self.Qf = [Cofunction(W.dual()) for _ in range(self.M)]
        self.Qf_fin = Cofunction(W.dual())


        if self.nonlinear_solver_parameters is None:
            # Use hybridised solver as default
            self.hybridised_solver = True
            alpha = self.Qdelta_imp[0, 0]/self.dt_coarse
            self.nonlinear_solver_parameters, self.appctx = hybridised_solver_parameters(self.base.equation, self.base.equation.field_names, alpha=alpha, tau_values=None, nonlinear=True)
        else:
            self.hybridised_solver = False
            self.appctx = None
        self.lag_rebuild_freq = None
        self._solver_call_count = 0
        if self.nonlinear_solver_parameters is not None:
            self.lag_rebuild_freq = self.nonlinear_solver_parameters.get("td_lag_rebuild", None)
        if self.lag_rebuild_freq is not None:
            if self.lag_rebuild_freq < 1:
                raise ValueError("SDC: td_lag_rebuild must be >= 1")
            elif not isinstance(self.lag_rebuild_freq, int):
                raise ValueError("SDC: td_lag_rebuild must be an integer")
            else:
                logger.info(f"RIDC: td_lag_rebuild set to {self.lag_rebuild_freq}. "
                            "Jacobian will be rebuilt every td_lag_rebuild solver calls.")


        # Set up a lag rebuild frequency for the Jacobian, if requested.
        # This is a timestep-level lag, not a sweep-level lag: the Jacobian is rebuilt every td_lag_rebuild timesteps
        self.lag_rebuild_freq = self.nonlinear_solver_parameters.get("td_lag_rebuild", None)
        if self.lag_rebuild_freq is not None:
            if self.lag_rebuild_freq < 1:
                raise ValueError("DeferredCorrection: td_lag_rebuild must be >= 1")
            elif not isinstance(self.lag_rebuild_freq, int):
                raise ValueError("DeferredCorrection: td_lag_rebuild must be an integer")
            else:
                logger.info(f"DeferredCorrection: td_lag_rebuild set to {self.lag_rebuild_freq}. "
                            "Jacobian will be rebuilt every td_lag_rebuild timesteps.")


        if self.qdelta_imp_type == "MIN-SR-FLEX":
            logger.info(
                "SDC: MIN-SR-FLEX diagonal is precomputed per (node, sweep) "
                "so that td_lag_rebuild can be honoured across timesteps; "
                "this builds maxk*M solver contexts instead of M.")
            # This gives a sweep dependent Qdelta of diag(nodes/k) for sweep k <= M and MIN-SR-S for sweep
            # k > M (Caklovic et al.).
            self._min_sr_flex_gen = MIN_SR_FLEX(
                nNodes=self.M, nodeType=self.node_type, quadType=self.quad_type)
            self.Qdelta_imp_diag_k = []
            for k in range(1, self.maxk + 1):
                diag_k = self._min_sr_flex_gen.computeQDelta(k=k)
                self.Qdelta_imp_diag_k.append(
                    [Constant(float(self.dt_coarse) * diag_k[m, m]) for m in range(self.M)])
        else:
            self.Qdelta_imp_diag = [Constant(self.Qdelta_imp[m, m]) for m in range(self.M)]

    @property
    def nlevels(self):
        return 1

    def Qf_form(self, m):
        """
        Weak (dual-space) form of node m's quadrature term:
        sum_i Q[m,i] * F(Unodes[i+1]), where F is the non-time-derivative
        part of the residual (implicit + explicit + source).
        """
        residual = None
        for i in range(self.M):
            w = float(self.Q[m, i])
            if w == 0.0:
                continue
            F = self.residual.label_map(
                lambda t: (not t.has_label(time_derivative)) and (not t.has_label(source_label)),
                replace_subject(self.Unodes[i+1], old_idx=self.idx),
                drop)
            F = F.label_map(all_terms, lambda t: Constant(w)*t)
            residual = F if residual is None else residual + F
        return residual.form

    @cached_property
    def Qf_assemblers(self):
        """
        Cached assemblers for each node's Qf_form, built once.
        """
        return [get_assembler(self.Qf_form(m), tensor=self.Qf[m]) for m in range(self.M)]

    def compute_Qf(self):
        """Assembles self.Qf[m] for every node m, once per sweep, before
        the per-node solves."""
        for m in range(self.M):
            self.Qf_assemblers[m].assemble(tensor=self.Qf[m])

    def Qf_fin_form(self):
        """
        As Qf_form, but for the final-update residual
        """
        residual = None
        for i in range(self.M):
            w = float(self.Qfin[i])
            if w == 0.0:
                continue
            F = self.residual.label_map(
                lambda t: (not t.has_label(time_derivative)) and (not t.has_label(source_label)),
                replace_subject(self.Unodes1[i+1], old_idx=self.idx),
                drop)
            F = F.label_map(all_terms, lambda t: Constant(w)*t)
            residual = F if residual is None else residual + F
        return residual.form

    @cached_property
    def Qf_fin_assembler(self):
        """Cached assembler for Qf_fin_form; built once."""
        return get_assembler(self.Qf_fin_form(), tensor=self.Qf_fin)

    def compute_Qf_fin(self):
        """Assemble self.Qf_fin, once, before the final-update solve."""
        self.Qf_fin_assembler.assemble(tensor=self.Qf_fin)


    def resval(self, m, k=None):
        """
        Set up the discretisation's residual for a given node m.

        Args:
            m (int): quadrature node index.
            k (int, optional): sweep number (1-indexed). Only used for
                MIN-SR-FLEX, to select the fixed per-(node, sweep) diagonal
                Constant built in setup(). Ignored for all other Qdelta_imp
                choices, which use a single per-node Constant.
        """
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

        # sum(j=1,M) Qdelta_exp[m,j]*(S(y_j^(k+1)) - S(y_j^k)), for
        # nonzero Qdelta_exp entries only.
        for i in range(self.M):

            Q_source = self.residual.label_map(
                lambda t: t.has_label(source_label),
                map_if_true=replace_subject(self.source_Uk[i+1], old_idx=self.idx),
                map_if_false=drop)
            Q_source = Q_source.label_map(
                all_terms,
                lambda t: Constant(self.Q[m, i])*t)
            residual += Q_source

            if self.Qdelta_exp[m, i] == 0:
                continue
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

            # Source terms
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
        diag_m = (self.Qdelta_imp_diag_k[k-1][m]
                  if self.qdelta_imp_type == "MIN-SR-FLEX"
                  else self.Qdelta_imp_diag[m])
        r_imp_kp1 = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.U_DC, old_idx=self.idx),
            map_if_false=drop)
        r_imp_kp1 = r_imp_kp1.label_map(
            all_terms,
            lambda t: diag_m*t)
        residual += r_imp_kp1
        r_imp_k = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.Unodes[m+1], old_idx=self.idx),
            map_if_false=drop)
        r_imp_k = r_imp_k.label_map(
            all_terms,
            lambda t: diag_m*t)
        residual -= r_imp_k

        return residual.form

    def _build_solver(self, m, k=None):
        """
        Build a single NonlinearVariationalSolver for node m (and, for
        MIN-SR-FLEX, sweep k).

        F = resval(m, k) + Qf[m]; J = derivative(resval(m, k), U_DC).

        Args:
            m (int): quadrature node index.
            k (int, optional): sweep number (1-indexed). Only relevant for
                MIN-SR-FLEX; ignored (and omitted from the solver name)
                otherwise.
        """
        Fval = self.resval(m, k)
        J = derivative(Fval, self.U_DC)
        F = Fval + self.Qf[m]
        problem = NonlinearVariationalProblem(F, self.U_DC, bcs=self.bcs, J=J)
        suffix = f"{m}_k{k}" if k is not None else f"{m}"
        solver_name = self.field_name + self.__class__.__name__ + suffix
        if self.hybridised_solver:
            # Use hybridised solver as default
            alpha = self.Qdelta_imp[m, m]/self.dt_coarse
            self.nonlinear_solver_parameters, self.appctx = hybridised_solver_parameters(self.equation, self.equation.field_names, alpha=alpha, tau_values=None, nonlinear=True)
        if self.lag_rebuild_freq is not None:
            problem._constant_jacobian = True
        return NonlinearVariationalSolver(
            problem, solver_parameters=self.nonlinear_solver_parameters, appctx=self.appctx,
            options_prefix=solver_name)

    @cached_property
    def solvers(self):
        """
        Solvers for each quadrature node (and, for MIN-SR-FLEX, each
        sweep). For MIN-SR-FLEX this returns a list-of-lists indexed
        [k-1][m], since the diagonal Qdelta_imp entry depends on the sweep
        number k; for all other Qdelta_imp choices it returns a flat list
        indexed [m].
        """
        if self.qdelta_imp_type == "MIN-SR-FLEX":
            return [[self._build_solver(m, k) for m in range(self.M)]
                    for k in range(1, self.maxk + 1)]
        return [self._build_solver(m) for m in range(self.M)]


    def resval_fin(self):
        """Set up the residual for final solve."""
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.U_fin, old_idx=self.idx),
                                    drop)
        F_exp = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                        replace_subject(self.Un, old_idx=self.idx),
                                        drop)
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)
        residual = a + F_exp
        for i in range(self.M):
            Q_source = self.residual.label_map(
                lambda t: t.has_label(source_label),
                map_if_true=replace_subject(self.source_Uk[i+1], old_idx=self.idx),
                map_if_false=drop)
            Q_source = Q_source.label_map(
                all_terms,
                lambda t: Constant(self.Qfin[i])*t)
            residual += Q_source
        return residual.form

    @cached_property
    def solver_fin(self):
        """Set up the problem and the solver for final update."""
        Fval = self.resval_fin()
        J = derivative(Fval, self.U_fin)
        F = Fval + self.Qf_fin
        problem = NonlinearVariationalProblem(F, self.U_fin, bcs=self.bcs, J=J)
        solver_name = self.field_name+self.__class__.__name__+"_final"
        return NonlinearVariationalSolver(problem, solver_parameters=self.linear_solver_parameters,
                                          options_prefix=solver_name)

    def _lag_reset(self, solvers):
        """Chooses whether to rebuild the Jacobian based on the lag frequency."""
        if self.lag_rebuild_freq is None:
            return
        rebuild = ((self._step_count - 1) % self.lag_rebuild_freq == 0)

        # For MIN-SR-FLEX, `solvers` is a list-of-lists indexed [k-1][m];
        # flatten it so every (node, sweep) solver context gets the same
        # rebuild/reuse decision for this timestep.
        flat_solvers = (
            [s for group in solvers for s in group]
            if self.qdelta_imp_type == "MIN-SR-FLEX" else solvers)

        for s in flat_solvers:
            s._ctx._jacobian_assembled = not rebuild

    @wrapper_apply
    def apply(self, x_out, x_in):
        self.Un.assign(x_in)
        self.U_start.assign(self.Un)
        solver_list = self.solvers

        self._lag_reset(solver_list)

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
                evaluate(self.Unodes[m], self.dt_coarse, x_out=self.source_Uk[m])

        # Iterate through correction sweeps
        k = 0
        while k < self.maxk:
            k += 1

            # On the final sweep with diagonal Qdelta and no final update, only the last
            # node is needed so only assemble Qf for that node.
            final_sweep_skip = (k == self.maxk and not self.final_update
                                and self._final_sweep_shortcut)
            if final_sweep_skip:
                self.Qf_assemblers[self.M-1].assemble(tensor=self.Qf[self.M-1])
            else:
                self.compute_Qf()

            # Loop through quadrature nodes and solve
            self.Unodes1[0].assign(self.Unodes[0])
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[0], self.dt_coarse, x_out=self.source_Uk[0])
            node_range = range(self.M, self.M+1) if final_sweep_skip else range(1, self.M+1)
            for m in node_range:
                # Set initial guess for solver, and pick correct solver
                if (self.formulation == "N2N"):
                    self.U_start.assign(self.Unodes1[m-1])
                # MIN-SR-FLEX has a dedicated solver per (node, sweep),
                # since its diagonal Qdelta_imp entry depends on k; all
                # other Qdelta_imp choices use a single solver per node.
                self.solver = (solver_list[k-1][m-1]
                               if self.qdelta_imp_type == "MIN-SR-FLEX"
                               else solver_list[m-1])
                self.U_DC.assign(self.Unodes[m])

                # Set sweep dependent solver tolerances if requested
                if self.sweep_tols is not None:
                    tol = self.sweep_tols[k-1]

                    self.solver.snes.ksp.setTolerances(
                        atol=tol["ksp_atol"],
                        rtol=tol["ksp_rtol"]
                    )

                    self.solver.snes.setTolerances(
                        atol=tol["snes_atol"],
                        rtol=tol["snes_rtol"]
                    )
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

                # Update iteration counters
                self.total_ksp_its += self.solver.snes.getLinearSolveIterations()
                self.total_snes_its += self.solver.snes.getIterationNumber()

                # Evaluate source terms
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes1[m], self.dt_coarse, x_out=self.source_Ukp1[m])

                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m])
            for m in range(1, self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])
                self.source_Uk[m].assign(self.source_Ukp1[m])

        if self.maxk > 0:
            # Compute value at dt rather than final quadrature node tau_M
            if self.final_update:
                # Compute final update quadrature term Qf_fin
                self.compute_Qf_fin()
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

        self._step_count = self._step_count + 1

class RIDC(object, metaclass=ABCMeta):
    """Class for Revisionist Integral Deferred Correction schemes."""

    def __init__(self, base_scheme, domain, M, K, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, reduced=True):
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
            reduced (bool, optional): whether to use reduced stencils for RIDC. Defaults to True.
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
        self.base.dt = float(self.dt_coarse)/(self.M)

        if reduced:
            self.Q = []
            for l in range(1, self.K + 1):
                _, _, Q = genQCoeffs(
                    "Collocation",
                    nNodes=l + 1,
                    nodeType="EQUID",
                    quadType="LOBATTO",
                    form="N2N"
                )
                Q = l * float(self.dt) * Q
                self.Q.append(Q)
        else:
            # Get integration weights
            _, _, self.Q = genQCoeffs(
                "Collocation",
                nNodes=self.K + 1,
                nodeType="EQUID",
                quadType="LOBATTO",
                form="N2N"
            )
            self.Q = self.K * float(self.dt) * self.Q

        # Set default linear and nonlinear solver options if none passed in
        if linear_solver_parameters is None:
            self.linear_solver_parameters = {'snes_type': 'ksponly',
                                             'ksp_type': 'cg',
                                             'pc_type': 'bjacobi',
                                             'sub_pc_type': 'ilu'}
        else:
            self.linear_solver_parameters = linear_solver_parameters
        
        self.nonlinear_solver_parameters = nonlinear_solver_parameters

        if self.nonlinear_solver_parameters is None:
            # Use hybridised solver as default
            self.hybridised_solver = True
            alpha = self.Q[0][0, 0]/self.dt_coarse
            self.nonlinear_solver_parameters, self.appctx = hybridised_solver_parameters(self.base.equation, self.base.equation.field_names, alpha=alpha, tau_values=None, nonlinear=True)
        else:
            self.hybridised_solver = False
            self.appctx = None
        self.lag_rebuild_freq = None
        self._solver_call_count = 0
        if self.nonlinear_solver_parameters is not None:
            self.lag_rebuild_freq = self.nonlinear_solver_parameters.get("td_lag_rebuild", None)
        if self.lag_rebuild_freq is not None:
            if self.lag_rebuild_freq < 1:
                raise ValueError("RIDC: td_lag_rebuild must be >= 1")
            elif not isinstance(self.lag_rebuild_freq, int):
                raise ValueError("RIDC: td_lag_rebuild must be an integer")
            else:
                logger.info(f"RIDC: td_lag_rebuild set to {self.lag_rebuild_freq}. "
                            "Jacobian will be rebuilt every td_lag_rebuild solver calls.")



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
        self.source_Uk = [Function(W) for _ in range(self.M+1)]
        self.source_Ukp1 = [Function(W) for _ in range(self.M+1)]
        self.U_DC = Function(W)
        self.U_start = Function(W)
        self.Un = Function(W)
        self.U_fin = Function(W)
        self.b = Cofunction(W.dual())
        self.source_Ukp1_m = Function(W)
        self.source_Uk_m = Function(W)
        self.Uk_mp1 = Function(W)
        self.Uk_m = Function(W)
        self.Ukp1_m = Function(W)

    @property
    def nlevels(self):
        return 1

    def _quad_weights_and_indices(self, k, m):
        """
        Returns (weight, fUnodes index) pairs for the RIDC quadrature term.

        Args:
            k (int): 1-based sweep index.
            m (int): 0-based local node index in the correction loop.
        """
        if self.reduced:
            Q = self.Q[k-1]
            if m < k:
                return [(float(Q[m+1, j]), j) for j in range(np.shape(Q)[1])]
            l = np.shape(Q)[0] - 1
            start = (m + 1) - l
            return [(float(Q[-1, j]), start + j) for j in range(l+1)]

        Q = self.Q
        if m < self.K:
            return [(float(Q[m+1, j]), j) for j in range(np.shape(Q)[1])]
        l = self.K
        start = (m + 1) - l
        return [(float(Q[-1, j]), start + j) for j in range(l+1)]

   
    def resval(self):
        """
        Jacobian-bearing part of the residual, evaluated at the unknown
        U_DC: the mass term plus dt*F(U_DC). Everything else is known data
        and goes into self.b via rhs_form/rhs_assemblers -- valid because
        RIDC's implicit diagonal coefficient is always the constant self.dt,
        independent of node or sweep (same structure as
        IMEXRungeKutta.resval() in the constant-diagonal case).
        """
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative), map_if_false=drop)
        residual = mass_form.label_map(
            all_terms, map_if_true=replace_subject(self.U_DC, old_idx=self.idx))

        r_imp = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.U_DC, old_idx=self.idx),
            map_if_false=drop)
        r_imp = r_imp.label_map(all_terms, lambda t: Constant(self.dt) * t)
        residual += r_imp

        return residual.form

    def rhs_form(self, k, m):
        """
        Dual-space RHS for sweep k, node m (0-indexed local node; corrects
        to Unodes1[m+1]). Combines every term that does NOT depend on the
        unknown U_DC:
        -mass(U_start)                                  (y_(m-1)^(k+1))
        -dt*F(y_m^k)                                     (previous sweep, node m+1)
        +dt*(S(y_(m-1)^(k+1)) - S(y_(m-1)^k))             explicit correction
        +dt*(source(y_(m-1)^(k+1)) - source(y_(m-1)^k))   source correction
        +sum_j Q[m+1,j]*(F+S)(y_j^k)                      dense RIDC
                                                            quadrature term,
                                                            evaluated directly
                                                            on the raw nodal
                                                            Functions -- no
                                                            mass-matrix solve.
        """
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative), map_if_false=drop)
        residual = -mass_form.label_map(
            all_terms, map_if_true=replace_subject(self.U_start, old_idx=self.idx))

        r_imp_k = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.Uk_mp1, old_idx=self.idx),
            map_if_false=drop)
        r_imp_k = r_imp_k.label_map(all_terms, lambda t: Constant(self.dt) * t)
        residual -= r_imp_k

        r_exp_kp1 = self.residual.label_map(
            lambda t: t.has_label(explicit),
            map_if_true=replace_subject(self.Ukp1_m, old_idx=self.idx),
            map_if_false=drop)
        r_exp_kp1 = r_exp_kp1.label_map(all_terms, lambda t: Constant(self.dt) * t)
        residual += r_exp_kp1
        r_exp_k = self.residual.label_map(
            lambda t: t.has_label(explicit),
            map_if_true=replace_subject(self.Uk_m, old_idx=self.idx),
            map_if_false=drop)
        r_exp_k = r_exp_k.label_map(all_terms, lambda t: Constant(self.dt) * t)
        residual -= r_exp_k

        r_source_kp1 = self.residual.label_map(
            lambda t: t.has_label(source_label),
            map_if_true=replace_subject(self.source_Ukp1_m, old_idx=self.idx),
            map_if_false=drop)
        r_source_kp1 = r_source_kp1.label_map(all_terms, lambda t: Constant(self.dt) * t)
        residual += r_source_kp1
        r_source_k = self.residual.label_map(
            lambda t: t.has_label(source_label),
            map_if_true=replace_subject(self.source_Uk_m, old_idx=self.idx),
            map_if_false=drop)
        r_source_k = r_source_k.label_map(all_terms, lambda t: Constant(self.dt) * t)
        residual -= r_source_k

        for w, idx in self._quad_weights_and_indices(k, m):
            if w == 0.0:
                continue
            Qi = self.residual.label_map(
                lambda t: t.has_label(implicit) or t.has_label(explicit),
                replace_subject(self.Unodes[idx], old_idx=self.idx),
                drop)
            Qi = Qi.label_map(all_terms, lambda t: Constant(w) * t)
            residual += Qi

            Qs = self.residual.label_map(
                lambda t: t.has_label(source_label),
                replace_subject(self.source_Uk[idx], old_idx=self.idx),
                drop)
            Qs = Qs.label_map(all_terms, lambda t: Constant(w) * t)
            residual += Qs

        return residual.form

    @cached_property
    def rhs_assemblers(self):
        """Cached assemblers for the RHS Cofunction self.b, indexed [k-1][m]."""
        return [[get_assembler(self.rhs_form(k, m), tensor=self.b)
                for m in range(self.M)]
                for k in range(1, self.K + 1)]

    @cached_property
    def solver(self):
        """
        Single shared solver -- valid because the implicit diagonal
        coefficient is always the constant self.dt, same as
        IMEXRungeKutta's shared-solver path for constant-diagonal tableaus.
        """
        F = self.resval() + self.b
        J = derivative(self.resval(), self.U_DC)
        problem = NonlinearVariationalProblem(F, self.U_DC, bcs=self.bcs, J=J)
        if self.lag_rebuild_freq is not None:
            problem._constant_jacobian = True
        solver_name = self.field_name + self.__class__.__name__
        return NonlinearVariationalSolver(
            problem, solver_parameters=self.nonlinear_solver_parameters,
            appctx=self.appctx, options_prefix=solver_name)

    def _lag_reset_solver(self):
        if self.lag_rebuild_freq is None:
            return
        rebuild = (self._solver_call_count % self.lag_rebuild_freq == 0)
        self.solver._ctx._jacobian_assembled = not rebuild

    def _lag_note_solver_call(self):
        if self.lag_rebuild_freq is None:
            return
        self._solver_call_count = self._solver_call_count + 1

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
        self.Unodes[0].assign(self.Un)
        self.M1 = self.K

        for m in range(self.M):
            self.base.dt = float(self.dt)
            self.base.apply(self.Unodes[m + 1], self.Unodes[m])

        for m in range(self.M + 1):
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[m], self.dt_coarse, x_out=self.source_Uk[m])

        for k in range(1, self.K + 1):
            self.Unodes1[0].assign(self.Unodes[0])
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[0], self.dt_coarse, x_out=self.source_Uk[0])
            if self.reduced:
                self.M1 = k

            for m in range(0, self.M1):
                self.rhs_assemblers[k - 1][m].assemble(tensor=self.b)

                self.U_start.assign(self.Unodes1[m])
                self.Ukp1_m.assign(self.Unodes1[m])
                self.Uk_mp1.assign(self.Unodes[m + 1])
                self.Uk_m.assign(self.Unodes[m])
                self.source_Ukp1_m.assign(self.source_Ukp1[m])
                self.source_Uk_m.assign(self.source_Uk[m])
                self.U_DC.assign(self.Unodes[m + 1])

                self._lag_reset_solver()
                self.solver.solve()
                self._lag_note_solver_call()
                self.Unodes1[m + 1].assign(self.U_DC)

                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes1[m + 1], self.dt_coarse, x_out=self.source_Ukp1[m + 1])

                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m + 1])

            for m in range(self.M1, self.M):
                self.rhs_assemblers[k - 1][m].assemble(tensor=self.b)

                self.U_start.assign(self.Unodes1[m])
                self.Ukp1_m.assign(self.Unodes1[m])
                self.Uk_mp1.assign(self.Unodes[m + 1])
                self.Uk_m.assign(self.Unodes[m])
                self.source_Ukp1_m.assign(self.source_Ukp1[m])
                self.source_Uk_m.assign(self.source_Uk[m])
                self.U_DC.assign(self.Unodes[m + 1])

                self._lag_reset_solver()
                self.solver.solve()
                self._lag_note_solver_call()
                self.Unodes1[m + 1].assign(self.U_DC)

                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes1[m + 1], self.dt_coarse, x_out=self.source_Ukp1[m + 1])

                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m + 1])

            for m in range(self.M + 1):
                self.Unodes[m].assign(self.Unodes1[m])
                self.source_Uk[m].assign(self.source_Ukp1[m])

        x_out.assign(self.Unodes[-1])