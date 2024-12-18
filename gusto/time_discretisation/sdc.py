u"""
Objects for discretising time derivatives using Spectral Deferred Correction
Methods.

SDC objects discretise ∂y/∂t = F(y), for variable y, time t and
operator F.

Written in Picard integral form this equation is
y(t) = y_n + int[t_n,t] F(y(s)) ds

Using some quadrature rule, we can evaluate y on a temporal quadrature node as
y_m = y_n + sum[j=1,M] q_mj*F(y_j)
where q_mj can be found by integrating Lagrange polynomials. This is similar to
how Runge-Kutta methods are formed.

In matrix form this equation is:
(I - dt*Q*F)(y)=y_n

Computing y by Picard iteration through k we get:
y^(k+1)=y^k + (y_n - (I - dt*Q*F)(y^k))

Finally, to get our SDC method we precondition this system, using some approximation
of Q Q_delta:
(I - dt*Q_delta*F)(y^(k+1)) = y_n + dt*(Q - Q_delta)F(y^k)

The zero-to-node (Z2N) formulation is then:
y_m^(k+1) = y_n + sum(j=1,M) q'_mj*(F(y_j^(k+1)) - F(y_j^k))
            + sum(j=1,M) q_mj*F(y_(m-1)^k)
for entires q_mj in Q and q'_mj in Q_delta.

Node-wise from previous quadrature node (N2N formulation), the implicit SDC calculation is:
y_m^(k+1) = y_(m-1)^(k+1) + dtau_m*(F(y_(m)^(k+1)) - F(y_(m)^k))
            + sum(j=1,M) s_mj*F(y_(m-1)^k)
where s_mj = q_mj - q_(m-1)j for entires q_ik in Q.


Key choices in our SDC method are:
- Choice of quadrature node type (e.g. gauss-lobatto)
- Number of quadrature nodes
- Number of iterations - each iteration increases the order of accuracy up to
  the order of the underlying quadrature
- Choice of Q_delta (e.g. Forward Euler, Backward Euler, LU-trick)
- How to get initial solution on quadrature nodes
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
from gusto.core.labels import (time_derivative, implicit, explicit)

from qmat import genQCoeffs, genQDeltaCoeffs

__all__ = ["SDC"]


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

        for t in self.residual:
            # Check all terms are labeled implicit or explicit
            if ((not t.has_label(implicit)) and (not t.has_label(explicit))
               and (not t.has_label(time_derivative))):
                raise NotImplementedError("Non time-derivative terms must be labeled as implicit or explicit")

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
        self.U_SDC = Function(W)
        self.U_start = Function(W)
        self.Un = Function(W)
        self.Q_ = Function(W)
        self.quad_final = Function(W)
        self.U_fin = Function(W)
        self.Urhs = Function(W)
        self.Uin = Function(W)

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
        L = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        residual_rhs = a - L
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
                                       map_if_true=replace_subject(self.U_SDC, old_idx=self.idx))
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

        # Add on final implicit terms
        # Qdelta_imp[m,m]*(F(y_(m)^(k+1)) - F(y_(m)^k))
        r_imp_kp1 = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.U_SDC, old_idx=self.idx),
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
            problem = NonlinearVariationalProblem(self.res(m), self.U_SDC, bcs=self.bcs)
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
                self.solver_rhs.solve()
                self.fUnodes[m-1].assign(self.Urhs)
            self.compute_quad()

            # Loop through quadrature nodes and solve
            self.Unodes1[0].assign(self.Unodes[0])
            for m in range(1, self.M+1):
                # Set Q or S matrix
                self.Q_.assign(self.quad[m-1])

                # Set initial guess for solver, and pick correct solver
                if (self.formulation == "N2N"):
                    self.U_start.assign(self.Unodes1[m-1])
                self.solver = solver_list[m-1]
                self.U_SDC.assign(self.Unodes[m])

                # Compute
                # for N2N:
                # y_m^(k+1) = y_(m-1)^(k+1) + dtau_m*(F(y_(m)^(k+1)) - F(y_(m)^k)
                #             + S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
                #             + sum(j=1,M) s_mj*(F+S)(y^k)
                # for Z2N:
                # y_m^(k+1) = y^n + sum(j=1,m) Qdelta_imp[m,j]*(F(y_(m)^(k+1)) - F(y_(m)^k))
                #             + sum(j=1,M)  Q_delta_exp[m,j]*(S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
                self.solver.solve()
                self.Unodes1[m].assign(self.U_SDC)

                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m])
            for m in range(1, self.M+1):
                self.Unodes[m].assign(self.Unodes1[m])

        if self.maxk > 0:
            # Compute value at dt rather than final quadrature node tau_M
            if self.final_update:
                for m in range(1, self.M+1):
                    self.Uin.assign(self.Unodes1[m])
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
