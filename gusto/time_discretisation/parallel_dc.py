"""
Objects for discretising time derivatives using time-parallel Deferred Correction
Methods.

This module inherits from the serial SDC and RIDC classes, and implements the
parallelisation of the SDC and RIDC methods using MPI.

SDC parallelises across the quadrature nodes by using diagonal QDelta matrices,
while RIDC parallelises across the correction iterations by using a reduced stencil
and pipelining.
"""

from firedrake import (
    Function, NonlinearVariationalProblem, NonlinearVariationalSolver, Constant
)
from firedrake.utils import cached_property
from gusto.time_discretisation.time_discretisation import wrapper_apply
from qmat import genQDeltaCoeffs
from gusto.time_discretisation.deferred_correction import SDC, RIDC
from gusto.core.logging import logger
from gusto.solvers.solver_presets import hybridised_solver_parameters
from firedrake.fml import (
    replace_subject, all_terms, drop, keep
)
from gusto.core.labels import (time_derivative, implicit, explicit, source_label)

__all__ = ["Parallel_RIDC", "Parallel_SDC"]


class Parallel_RIDC(RIDC):
    """Class for Parallel Revisionist Integral Deferred Correction schemes."""

    def __init__(self, base_scheme, domain, M, K, J, output_freq, flush_freq=None, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, communicator=None):
        """
        Initialise RIDC object
        Args:
            base_scheme (:class:`TimeDiscretisation`): Base time stepping scheme to get first guess of solution on
                quadrature nodes.
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            M (int): Number of subintervals
            K (int): Max number of correction interations
            J (int): Number of intervals
            output_freq (int): Frequency at which output is done
            flush_freq (int): Frequency at which to flush the pipeline
            field_name (str, optional): name of the field to be evolved.
                Defaults to None.
            linear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying linear solver. Defaults to None.
            nonlinear_solver_parameters (dict, optional): dictionary of parameters to
                pass to the underlying nonlinear solver. Defaults to None.
            limiter (:class:`Limiter` object, optional): a limiter to apply to
                the evolving field to enforce monotonicity. Defaults to None.
            communicator (MPI communicator, optional): communicator for parallel execution. Defaults to None.
        """

        super(Parallel_RIDC, self).__init__(base_scheme, domain, M, K, field_name,
                                            linear_solver_parameters, nonlinear_solver_parameters,
                                            limiter, reduced=True)
        self.comm = communicator
        self.TAG_EXCHANGE_FIELD = 11  # Tag for sending nodal fields (Firedrake Functions)
        self.TAG_EXCHANGE_SOURCE = self.TAG_EXCHANGE_FIELD + J  # Tag for sending nodal source fields (Firedrake Functions)
        self.TAG_FLUSH_PIPE = self.TAG_EXCHANGE_SOURCE + J  # Tag for flushing pipe and restarting
        self.TAG_FINAL_OUT = self.TAG_FLUSH_PIPE + J  # Tag for the final broadcast and output
        self.TAG_END_INTERVAL = self.TAG_FINAL_OUT + J  # Tag for telling the rank above you that you have ended interval j

        if flush_freq is None:
            self.flush_freq = 1
        else:
            self.flush_freq = flush_freq

        self.J = J
        self.step = 1
        self.output_freq = output_freq

        

        if self.flush_freq == 0 or (self.flush_freq != 0 and self.output_freq % self.flush_freq != 0):
            logger.warn("Output on all parallel in time ranks will not be the same until end of run!")

        # Checks for parallel RIDC
        if self.comm is None:
            raise ValueError("No communicator provided. Please provide a valid MPI communicator.")
        if self.comm.ensemble_comm.size != self.K + 1:
            raise ValueError("Number of ranks must be equal to K+1 for Parallel RIDC.")
        if self.M < self.K*(self.K+1)//2:
            raise ValueError("Number of subintervals M must be greater than K*(K+1)/2 for Parallel RIDC.")

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
        super(Parallel_RIDC, self).setup(equation, apply_bcs, *active_labels)

        self.Uk_mp1 = Function(self.W)
        self.Uk_m = Function(self.W)
        self.Ukp1_m = Function(self.W)
        self.U_send = [Function(self.W) for _ in range(self.M+1)]
        self.Uprev = Function(self.W)

    @wrapper_apply
    def apply(self, x_out, x_in):
        # Set up varibles on this rank
        x_out.assign(x_in)
        self.kval = self.comm.ensemble_comm.rank
        self.Un.assign(x_in)
        self.Unodes[0].assign(self.Un)
        # Loop through quadrature nodes and solve
        if (self.flush_freq > 0 and (self.step -1) % self.flush_freq == 0):
            # After a flush, predictor and corrector start from same point
            self.Unodes[0].assign(x_in)
        else:
            # No flush - predictor should be last timestep's pipeline value
            self.Unodes[0].assign(self.Uprev)
        self.Unodes1[0].assign(x_in)
        # for evaluate in self.evaluate_source:
        #     evaluate(self.Unodes[0], self.base.dt, x_out=self.source_Uk[0])
        self.Uin.assign(self.Unodes[0])
        self.solver_rhs.solve()
        self.fUnodes[0].assign(self.Urhs)

        # On first communicator, we do the predictor step
        if (self.comm.ensemble_comm.rank == 0):
            # Base timestepper
            for m in range(self.M):
                self.base.dt = float(self.dt)
                self.base.apply(self.Unodes[m+1], self.Unodes[m])
                # for evaluate in self.evaluate_source:
                #     evaluate(self.Unodes[m+1], self.base.dt, x_out=self.source_Uk[m+1])

                # Send base guess to k+1 correction
                self.U_send[m+1].assign(self.Unodes[m+1])
                self.comm.isend(self.U_send[m+1], dest=self.kval+1, tag=self.TAG_EXCHANGE_FIELD + self.step + (m+1)*100)
                # self.comm.send(self.source_Uk[m+1], dest=self.kval+1, tag=self.TAG_EXCHANGE_SOURCE + self.step)
        else:
            for m in range(1, self.kval + 1):
                # Receive and evaluate the stencil of guesses we need to correct
                self.comm.recv(self.U_send[m], source=self.kval-1, tag=self.TAG_EXCHANGE_FIELD + self.step + (m)*100)
                self.Unodes[m].assign(self.U_send[m])
                # self.comm.recv(self.source_Uk[m], source=self.kval-1, tag=self.TAG_EXCHANGE_SOURCE + self.step)
                self.Uin.assign(self.Unodes[m])
                # for evaluate in self.evaluate_source:
                #     evaluate(self.Uin, self.base.dt, x_out=self.source_in)
                self.solver_rhs.solve()
                self.fUnodes[m].assign(self.Urhs)
            for m in range(0, self.kval):
                # Set S matrix
                self.Q_.assign(self.compute_quad(self.Q[self.kval-1], self.fUnodes, m+1))

                # Set initial guess for solver, and pick correct solver
                self.U_start.assign(self.Unodes1[m])
                self.Ukp1_m.assign(self.Unodes1[m])
                self.Uk_mp1.assign(self.Unodes[m+1])
                self.Uk_m.assign(self.Unodes[m])
                self.source_Ukp1_m.assign(self.source_Ukp1[m])
                self.source_Uk_m.assign(self.source_Uk[m])
                self.U_DC.assign(self.Unodes[m+1])

                # Compute
                # y_m^(k+1) = y_(m-1)^(k+1) + dt*(F(y_(m)^(k+1)) - F(y_(m)^k)
                #             + S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
                #             + sum(j=1,M) s_mj*(F+S)(y_j^k)
                self.solver.solve()
                self.Unodes1[m+1].assign(self.U_DC)

                # Evaluate source terms
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes1[m+1], self.base.dt, x_out=self.source_Ukp1[m+1])

                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m+1])
                # Send our updated value to next communicator
                if self.kval < self.K:
                    self.U_send[m+1].assign(self.Unodes1[m+1])
                    self.comm.isend(self.U_send[m+1], dest=self.kval+1, tag=self.TAG_EXCHANGE_FIELD + self.step + (m+1)*100)
                    #self.comm.isend(self.source_Ukp1[m+1], dest=self.kval+1, tag=self.TAG_EXCHANGE_SOURCE + self.step)

            for m in range(self.kval, self.M):
                # Receive the guess we need to correct and evaluate the rhs
                self.comm.recv(self.U_send[m+1], source=self.kval-1, tag=self.TAG_EXCHANGE_FIELD + self.step + (m+1)*100)
                self.Unodes[m+1].assign(self.U_send[m+1])
                #self.comm.recv(self.source_Uk[m+1], source=self.kval-1, tag=self.TAG_EXCHANGE_SOURCE + self.step)
                self.Uin.assign(self.Unodes[m+1])
                # for evaluate in self.evaluate_source:
                #     evaluate(self.Uin, self.base.dt, x_out=self.source_in)
                self.solver_rhs.solve()
                self.fUnodes[m+1].assign(self.Urhs)

                # Set S matrix
                self.Q_.assign(self.compute_quad_final(self.Q[self.kval-1], self.fUnodes, m+1))

                # Set initial guess for solver, and pick correct solver
                self.U_start.assign(self.Unodes1[m])
                self.Ukp1_m.assign(self.Unodes1[m])
                self.Uk_mp1.assign(self.Unodes[m+1])
                self.Uk_m.assign(self.Unodes[m])
                self.source_Ukp1_m.assign(self.source_Ukp1[m])
                self.source_Uk_m.assign(self.source_Uk[m])
                self.U_DC.assign(self.Unodes[m+1])

                # y_m^(k+1) = y_(m-1)^(k+1) + dt*(F(y_(m)^(k+1)) - F(y_(m)^k)
                #             + S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
                #             + sum(j=1,M) s_mj*(F+S)(y^k)
                self.solver.solve()
                self.Unodes1[m+1].assign(self.U_DC)

                # # Evaluate source terms
                # for evaluate in self.evaluate_source:
                #     evaluate(self.Unodes1[m+1], self.base.dt, x_out=self.source_Ukp1[m+1])

                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m+1])

                # Send our updated value to next communicator
                if self.kval < self.K:
                    self.U_send[m+1].assign(self.Unodes1[m+1])
                    self.comm.isend(self.U_send[m+1], dest=self.kval+1, tag=self.TAG_EXCHANGE_FIELD + self.step + (m+1)*100)
                    #self.comm.isend(self.source_Ukp1[m+1], dest=self.kval+1, tag=self.TAG_EXCHANGE_SOURCE + self.step)

            # for m in range(self.M+1):
            #     self.Unodes[m].assign(self.Unodes1[m])
            #     self.source_Uk[m].assign(self.source_Ukp1[m])

        if (self.flush_freq > 0 and self.step % self.flush_freq == 0) or self.step == self.J:
            # Flush the pipe to ensure all ranks have the same data
            if (self.kval == self.K):
                x_out.assign(self.Unodes1[-1])
                for i in range(self.K):
                    self.comm.isend(x_out, dest=i, tag=self.TAG_FLUSH_PIPE + self.step)
            else:
                self.comm.recv(x_out, source=self.K, tag=self.TAG_FLUSH_PIPE + self.step)
        else:
            if self.kval == 0:
                x_out.assign(self.Unodes[-1])
            else:
                x_out.assign(self.Unodes1[-1])

        self.Uprev.assign(self.Unodes[-1])

        self.step += 1


class Parallel_SDC(SDC):
    """Class for Spectral Deferred Correction schemes."""

    def __init__(self, base_scheme, domain, M, maxk, quad_type, node_type, qdelta_imp, qdelta_exp,
                 field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None, final_update=True,
                 limiter=None, options=None, initial_guess="base", communicator=None, exp_base_scheme=None):
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
            communicator (MPI communicator, optional): communicator for parallel execution. Defaults to None.
        """
        super().__init__(base_scheme, domain, M, maxk, quad_type, node_type, qdelta_imp, qdelta_exp,
                         formulation="Z2N", field_name=field_name,
                         linear_solver_parameters=linear_solver_parameters, nonlinear_solver_parameters=nonlinear_solver_parameters,
                         final_update=final_update,
                         limiter=limiter, initial_guess=initial_guess)
        self.comm = communicator
        self.alpha = Constant(0.0)  # Initial value, will be updated in solver setup
        # Checks for parallel SDC
        if self.comm is None:
            raise ValueError("No communicator provided. Please provide a valid MPI communicator.")
        if self.comm.ensemble_comm.size != self.M:
            raise ValueError("Number of ranks must be equal to the number of nodes M for Parallel SDC.")
        self.exp_base = exp_base_scheme
        
    def setup(self, equation, apply_bcs=True, *active_labels):
        """
        Set up the SDC time discretisation based on the equation.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            apply_bcs (bool, optional): whether to apply the equation's boundary
                conditions. Defaults to True.
            *active_labels (:class:`Label`): labels indicating which terms of
                the equation are active. Defaults to all terms.
        """
        super(Parallel_SDC, self).setup(equation, apply_bcs, *active_labels)

        if self.qdelta_imp_type == "MIN-SR-FLEX":
            _ = self.solvers
        else:
            _ = self.solver

        self.Unodes = [Function(self.W) for _ in range(self.M+1)]

        if self.exp_base is not None:
            exp_eqn = equation.label_map(lambda t: t.has_label(implicit), map_if_true=drop, map_if_false=keep)
            self.exp_base.setup(exp_eqn, apply_bcs, *active_labels)

    def compute_quad(self):
        """
        Computes integration of F(y) on quadrature nodes
        """
        x = Function(self.W)
        for j in range(self.M):
            x.assign(float(self.Q[j, self.comm.ensemble_comm.rank])*self.fUnodes[self.comm.ensemble_comm.rank])
            self.comm.reduce(x, self.quad[j], root=j)

    def compute_quad_final(self):
        """
        Computes final integration of F(y) on quadrature nodes
        """
        x = Function(self.W)
        x.assign(float(self.Qfin[self.comm.ensemble_comm.rank])*self.fUnodes[self.comm.ensemble_comm.rank])
        self.comm.allreduce(x, self.quad_final)

    def res_k(self, k):
        """Set up the discretisation's residual for a given node m."""
        # Add time derivative terms  y^(k+1)_m - y_start for node m. y_start is y_n for Z2N formulation
        # and y^(k)_m for N2N formulation
        m = self.comm.ensemble_comm.rank
        dt = float(self.dt_coarse)
        all_QD = genQDeltaCoeffs(
                    self.qdelta_imp_type,
                    nSweeps=self.maxk,
                    form=self.formulation,
                    nodes=self.nodes / dt,
                    Q=self.Q / dt,
                    nNodes=self.M,
                    nodeType=self.node_type,
                    quadType=self.quad_type,
                )
        qd_dt = all_QD[k-1][m, m]*dt
        logger.info(f"MIN-SR-FLEX RES: M={self.M}, rank={self.comm.ensemble_comm.rank}, k={k}, qd={all_QD[k-1][m, m]:.6e}, qd_dt={qd_dt:.6e}")
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)
        residual = mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.U_DC, old_idx=self.idx))
        residual -= mass_form.label_map(all_terms,
                                        map_if_true=replace_subject(self.U_start, old_idx=self.idx))
        # # Loop through nodes up to m-1 and calcualte
        # # sum(j=1,m-1) Qdelta_imp[m,j]*(F(y_(m)^(k+1)) - F(y_(m)^k))
        # for i in range(m):
        #     r_imp_kp1 = self.residual.label_map(
        #         lambda t: t.has_label(implicit),
        #         map_if_true=replace_subject(self.Unodes1[i+1], old_idx=self.idx),
        #         map_if_false=drop)
        #     r_imp_kp1 = r_imp_kp1.label_map(
        #         all_terms,
        #         lambda t: Constant(self.Qdelta_imp[m, i])*t)
        #     residual += r_imp_kp1
        #     r_imp_k = self.residual.label_map(
        #         lambda t: t.has_label(implicit),
        #         map_if_true=replace_subject(self.Unodes[i+1], old_idx=self.idx),
        #         map_if_false=drop)
        #     r_imp_k = r_imp_k.label_map(
        #         all_terms,
        #         lambda t: Constant(self.Qdelta_imp[m, i])*t)
        #     residual -= r_imp_k
        # # Loop through nodes up to m-1 and calcualte
        # #  sum(j=1,M)  Q_delta_exp[m,j]*(S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
        # for i in range(self.M):
        #     r_exp_kp1 = self.residual.label_map(
        #         lambda t: t.has_label(explicit),
        #         map_if_true=replace_subject(self.Unodes1[i+1], old_idx=self.idx),
        #         map_if_false=drop)
        #     r_exp_kp1 = r_exp_kp1.label_map(
        #         all_terms,
        #         lambda t: Constant(self.Qdelta_exp[m, i])*t)

        #     residual += r_exp_kp1
        #     r_exp_k = self.residual.label_map(
        #         lambda t: t.has_label(explicit),
        #         map_if_true=replace_subject(self.Unodes[i+1], old_idx=self.idx),
        #         map_if_false=drop)
        #     r_exp_k = r_exp_k.label_map(
        #         all_terms,
        #         lambda t: Constant(self.Qdelta_exp[m, i])*t)
        #     residual -= r_exp_k

        #     # Calculate source terms
        #     r_source_kp1 = self.residual.label_map(
        #         lambda t: t.has_label(source_label),
        #         map_if_true=replace_subject(self.source_Ukp1[i+1], old_idx=self.idx),
        #         map_if_false=drop)
        #     r_source_kp1 = r_source_kp1.label_map(
        #         all_terms,
        #         lambda t: Constant(self.Qdelta_exp[m, i])*t)
        #     residual += r_source_kp1

        #     r_source_k = self.residual.label_map(
        #         lambda t: t.has_label(source_label),
        #         map_if_true=replace_subject(self.source_Uk[i+1], old_idx=self.idx),
        #         map_if_false=drop)
        #     r_source_k = r_source_k.label_map(
        #         all_terms,
        #         map_if_true=lambda t: Constant(self.Qdelta_exp[m, i])*t)
        #     residual -= r_source_k

        # Add on final implicit terms
        # Qdelta_imp[m,m]*(F(y_(m)^(k+1)) - F(y_(m)^k))
        r_imp_kp1 = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.U_DC, old_idx=self.idx),
            map_if_false=drop)
        r_imp_kp1 = r_imp_kp1.label_map(
            all_terms,
            lambda t: Constant(qd_dt)*t)
        residual += r_imp_kp1
        r_imp_k = self.residual.label_map(
            lambda t: t.has_label(implicit),
            map_if_true=replace_subject(self.Unodes[m+1], old_idx=self.idx),
            map_if_false=drop)
        r_imp_k = r_imp_k.label_map(
            all_terms,
            lambda t: Constant(qd_dt)*t)
        residual -= r_imp_k

        # Add on error term. sum(j=1,M) q_mj*F(y_m^k) for Z2N formulation
        # and sum(j=1,M) s_mj*F(y_m^k) for N2N formulation, where s_mj = q_mj-q_m-1j
        # and s1j = q1j.
        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_, old_idx=self.idx),
                                    drop)
        residual += Q
        return residual.form

    @property
    def res_rhs_imp(self):
        """Set up the residual for the calculation of F(y)."""
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
        # F(y)
        L = self.residual.label_map(lambda t: any(t.has_label(time_derivative, source_label, explicit)),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        L_source = self.residual.label_map(lambda t: t.has_label(source_label),
                                           replace_subject(self.source_in, old_idx=self.idx),
                                           drop)
        residual_rhs = a - (L + L_source)
        return residual_rhs.form

    @property
    def res_rhs_exp(self):
        """Set up the residual for the calculation of F(y)."""
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
        # F(y)
        L = self.residual.label_map(lambda t: any(t.has_label(time_derivative, source_label, implicit)),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        L_source = self.residual.label_map(lambda t: t.has_label(source_label),
                                           replace_subject(self.source_in, old_idx=self.idx),
                                           drop)
        residual_rhs = a - (L + L_source)
        return residual_rhs.form


    @cached_property
    def solver_rhs_imp(self):
        """Set up the problem and the solver for mass matrix inversion."""
        # setup linear solver using rhs residual defined in derived class
        prob_rhs = NonlinearVariationalProblem(self.res_rhs_imp, self.Urhs, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_rhs_imp"
        return NonlinearVariationalSolver(prob_rhs, solver_parameters=self.linear_solver_parameters,
                                          options_prefix=solver_name)
    

    @cached_property
    def solver_rhs_exp(self):
        """Set up the problem and the solver for mass matrix inversion."""
        # setup linear solver using rhs residual defined in derived class
        prob_rhs = NonlinearVariationalProblem(self.res_rhs_exp, self.Urhs, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_rhs_exp"
        return NonlinearVariationalSolver(prob_rhs, solver_parameters=self.linear_solver_parameters,
                                          options_prefix=solver_name)

    @cached_property
    def solver(self):
        """Set up a list of solvers for each problem at a node m."""
        m = self.comm.ensemble_comm.rank
        # setup solver using residual defined in derived class
        self.alpha.assign(self.Qdelta_imp[m, m]/float(self.dt_coarse))
        dt = float(self.dt_coarse)
        QD_imp = genQDeltaCoeffs(
                    self.qdelta_imp_type,
                    form=self.formulation,
                    nodes=self.nodes/dt,
                    Q=self.Q/dt,
                    nNodes=self.M,
                    nodeType=self.node_type,
                    quadType=self.quad_type
                )
        alpha = QD_imp[m, m]
        #print("Setting up hybridised solver with alpha = %s" % alpha)
        if self.nonlinear_solver_parameters is None:
            self.nonlinear_solver_parameters, self.appctx = hybridised_solver_parameters(self.equation, self.equation.field_names, alpha=alpha, tau_values=None, nonlinear=True, imex=True)
        else:
            self.appctx = None
        problem = NonlinearVariationalProblem(self.res(m), self.U_DC, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__ + "%s" % (m)
        solver = NonlinearVariationalSolver(problem, solver_parameters=self.nonlinear_solver_parameters, appctx=self.appctx, options_prefix=solver_name)
        return solver
    
    @cached_property
    def solvers(self):
        """Set up a list of solvers for each problem at a node m."""
        solvers = []
        dt = float(self.dt_coarse)
        all_Qdelta = genQDeltaCoeffs(
            self.qdelta_imp_type,
            nSweeps=self.maxk,
            form=self.formulation,
            nodes=self.nodes / dt,
            Q=self.Q / dt,
            nNodes=self.M,
            nodeType=self.node_type,
            quadType=self.quad_type,
        )
        for k in range(1, self.maxk+1):
            Qdelta_imp_k = all_Qdelta[k-1]
            alpha_k = Qdelta_imp_k[self.comm.ensemble_comm.rank, self.comm.ensemble_comm.rank]
            logger.info(f"MIN-SR-FLEX: M={self.M}, rank={self.comm.ensemble_comm.rank}, k={k}, alpha_k={alpha_k:.6e}, qd_dt={alpha_k*dt:.6e}")
            if self.nonlinear_solver_parameters is None:
                nonlinear_solver_parameters_k, appctx_k = hybridised_solver_parameters(self.equation, self.equation.field_names, alpha=alpha_k, tau_values=None, nonlinear=True, imex=True)
            else:
                nonlinear_solver_parameters_k = self.nonlinear_solver_parameters
                appctx_k = None
            problem_k = NonlinearVariationalProblem(self.res_k(k), self.U_DC, bcs=self.bcs)
            solver_name_k = self.field_name+self.__class__.__name__ + "%s_k%s" % (self.comm.ensemble_comm.rank, k)
            solver_k = NonlinearVariationalSolver(problem_k, solver_parameters=nonlinear_solver_parameters_k, appctx=appctx_k, options_prefix=solver_name_k)
            solvers.append(solver_k)
        return solvers

    @wrapper_apply
    def apply(self, x_out, x_in):
        self.Un.assign(x_in)
        self.U_start.assign(self.Un)
        solver_list = self.solvers

        # Compute initial guess on quadrature nodes with low-order
        # base timestepper
        self.Unodes[0].assign(self.Un)
        self.Unodes_exp[0].assign(self.Un)
        if (self.base_flag):
            for m in range(self.M):
                self.base.dt = float(self.dtau[m])
                self.base.apply(self.Unodes[m+1], self.Unodes[m])
        elif (self.exp_base is not None):
            for m in range(self.M):
                self.exp_base.dt = float(self.dtau[m])
                self.exp_base.apply(self.Unodes_exp[m+1], self.Unodes_exp[m])
                self.Unodes[m+1].assign(self.Un)
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
                # self.Qdelta_imp = float(self.dt_coarse)*genQDeltaCoeffs(
                #     self.qdelta_imp_type,
                #     form=self.formulation,
                #     nodes=self.nodes,
                #     Q=self.Q,
                #     nNodes=self.M,
                #     nodeType=self.node_type,
                #     quadType=self.quad_type,
                #     k=k
                # )
                # self.alpha.assign(self.Qdelta_imp[self.comm.ensemble_comm.rank, self.comm.ensemble_comm.rank]/float(self.dt_coarse))
                solver = solver_list[k-1]
            else:
                solver = self.solver

            # Compute for N2N: sum(j=1,M) (s_mj*F(y_m^k) +  s_mj*S(y_m^k))
            # for Z2N: sum(j=1,M) (q_mj*F(y_m^k) +  q_mj*S(y_m^k))

            # # Include source terms
            # for evaluate in self.evaluate_source:
            #     evaluate(self.Uin, self.base.dt, x_out=self.source_in)
            if k ==1:
                self.Uin.assign(self.Unodes_exp[self.comm.ensemble_comm.rank+1])
                self.solver_rhs_exp.solve()
                self.fUnodes[self.comm.ensemble_comm.rank].assign(self.Urhs)
                self.Uin.assign(self.Unodes[self.comm.ensemble_comm.rank+1])
                self.solver_rhs_imp.solve()
                self.fUnodes[self.comm.ensemble_comm.rank].assign(self.fUnodes[self.comm.ensemble_comm.rank]+self.Urhs)
            else:
                self.Uin.assign(self.Unodes[self.comm.ensemble_comm.rank+1])
                self.solver_rhs.solve()
                self.fUnodes[self.comm.ensemble_comm.rank].assign(self.Urhs)

            self.compute_quad()

            # Loop through quadrature nodes and solve
            self.Unodes1[0].assign(self.Unodes[0])
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[0], self.base.dt, x_out=self.source_Uk[0])

            # Set Q or S matrix
            self.Q_.assign(self.quad[self.comm.ensemble_comm.rank])

            # Set initial guess for solver, and pick correct solver
            #self.solver = solver_list[self.comm.ensemble_comm.rank]
            self.U_DC.assign(self.Unodes[self.comm.ensemble_comm.rank+1])

            # Compute
            # for N2N:
            # y_m^(k+1) = y_(m-1)^(k+1) + dtau_m*(F(y_(m)^(k+1)) - F(y_(m)^k)
            #             + S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
            #             + sum(j=1,M) s_mj*(F+S)(y^k)
            # for Z2N:
            # y_m^(k+1) = y^n + sum(j=1,m) Qdelta_imp[m,j]*(F(y_(m)^(k+1)) - F(y_(m)^k))
            #             + sum(j=1,M)  Q_delta_exp[m,j]*(S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
            solver.solve()
            self.Unodes1[self.comm.ensemble_comm.rank+1].assign(self.U_DC)

            # Evaluate source terms
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes1[self.comm.ensemble_comm.rank+1], self.base.dt, x_out=self.source_Ukp1[self.comm.ensemble_comm.rank+1])

            # Apply limiter if required
            if self.limiter is not None:
                self.limiter.apply(self.Unodes1[self.comm.ensemble_comm.rank+1])

            self.Unodes[self.comm.ensemble_comm.rank+1].assign(self.Unodes1[self.comm.ensemble_comm.rank+1])
            self.source_Uk[self.comm.ensemble_comm.rank+1].assign(self.source_Ukp1[self.comm.ensemble_comm.rank+1])

        if self.maxk > 0:
            # Compute value at dt rather than final quadrature node tau_M
            if self.final_update:
                self.Uin.assign(self.Unodes1[self.comm.ensemble_comm.rank+1])
                self.source_in.assign(self.source_Ukp1[self.comm.ensemble_comm.rank+1])
                self.solver_rhs.solve()
                self.fUnodes[self.comm.ensemble_comm.rank].assign(self.Urhs)
                self.compute_quad_final()
                # Compute y_(n+1) = y_n + sum(j=1,M) q_j*F(y_j)
                if self.comm.ensemble_comm.rank == self.M-1:
                    self.U_fin.assign(self.Unodes[-1])
                self.comm.bcast(self.U_fin, self.M-1)
                self.solver_fin.solve()
                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.U_fin)
                x_out.assign(self.U_fin)
            else:
                # Take value at final quadrature node dtau_M
                if self.comm.ensemble_comm.rank == self.M-1:
                    x_out.assign(self.Unodes[-1])
                self.comm.bcast(x_out, self.M-1)
        else:
            # Take value at final quadrature node dtau_M
            if self.comm.ensemble_comm.rank == self.M-1:
                x_out.assign(self.Unodes[-1])
            self.comm.bcast(x_out, self.M-1)
