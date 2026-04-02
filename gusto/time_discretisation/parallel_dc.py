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
    Function, NonlinearVariationalProblem, NonlinearVariationalSolver, Constant, SpatialCoordinate, norm
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
from gusto.core.labels import (time_derivative, implicit, explicit, source_label, transporting_velocity)

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
        self.Uin.assign(self.Unodes[0])
        self.solver_rhs.solve()
        self.fUnodes[0].assign(self.Urhs)

        # On first communicator, we do the predictor step
        if (self.comm.ensemble_comm.rank == 0):
            # Base timestepper
            for m in range(self.M):
                self.base.dt = float(self.dt)
                self.base.apply(self.Unodes[m+1], self.Unodes[m])

                # Send base guess to k+1 correction
                self.U_send[m+1].assign(self.Unodes[m+1])
                self.comm.isend(self.U_send[m+1], dest=self.kval+1, tag=self.TAG_EXCHANGE_FIELD + self.step + (m+1)*100)
        else:
            for m in range(1, self.kval + 1):
                # Receive and evaluate the stencil of guesses we need to correct
                self.comm.recv(self.U_send[m], source=self.kval-1, tag=self.TAG_EXCHANGE_FIELD + self.step + (m)*100)
                self.Unodes[m].assign(self.U_send[m])
                self.Uin.assign(self.Unodes[m])
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

            for m in range(self.kval, self.M):
                # Receive the guess we need to correct and evaluate the rhs
                self.comm.recv(self.U_send[m+1], source=self.kval-1, tag=self.TAG_EXCHANGE_FIELD + self.step + (m+1)*100)
                self.Unodes[m+1].assign(self.U_send[m+1])
                self.Uin.assign(self.Unodes[m+1])
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

                self.solver.solve()
                self.Unodes1[m+1].assign(self.U_DC)

                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m+1])

                # Send our updated value to next communicator
                if self.kval < self.K:
                    self.U_send[m+1].assign(self.Unodes1[m+1])
                    self.comm.isend(self.U_send[m+1], dest=self.kval+1, tag=self.TAG_EXCHANGE_FIELD + self.step + (m+1)*100)

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
                 limiter=None, options=None, initial_guess="base", communicator=None, exp_base_scheme=None,
                 n_transport_subcycles=10):
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
            exp_base_scheme: unused, kept for API compatibility.
            n_transport_subcycles (int, optional): number of explicit subcycles for transport.
                Defaults to 3.
        """
        super().__init__(base_scheme, domain, M, maxk, quad_type, node_type, qdelta_imp, qdelta_exp,
                         formulation="Z2N", field_name=field_name,
                         linear_solver_parameters=linear_solver_parameters, nonlinear_solver_parameters=nonlinear_solver_parameters,
                         final_update=final_update,
                         limiter=limiter, initial_guess=initial_guess)
        self.comm = communicator
        self.alpha = Constant(0.0)  # Initial value, will be updated in solver setup
        self.n_transport_subcycles = n_transport_subcycles
        # Checks for parallel SDC
        if self.comm is None:
            raise ValueError("No communicator provided. Please provide a valid MPI communicator.")
        if self.comm.ensemble_comm.size != self.M:
            raise ValueError("Number of ranks must be equal to the number of nodes M for Parallel SDC.")
        # exp_base_scheme retained for API compatibility but transport is done via solver_rhs_exp
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

        self.Unodes_exp = [Function(self.W) for _ in range(self.M+1)]
        u_idx = self.equation.field_names.index('u')
        self.u_adv_avg = Function(self.W.subfunctions[u_idx])
        self.adv_inc = Function(self.W)
        # Reusable buffer for subcycled transport
        self.u_transport = Function(self.W)
        self.u_stage = Function(self.W)

    # def _transport_trajectory(self, n_sub, dtau_list):
    #     u_idx = self.equation.field_names.index('u')
    #     for m, dtau_m in enumerate(dtau_list):
    #         # Freeze transporting velocity from current nodal wind
    #         self.u_adv_avg.assign(self.Unodes[m].subfunctions[u_idx])
    #         dt_sub = float(dtau_m) / n_sub
    #         self.u_transport.assign(self.Unodes_exp[m])
    #         for _ in range(n_sub):
    #             self.Uin.assign(self.u_transport)
    #             self.solver_rhs_exp.solve()
    #             self.u_transport.assign(self.u_transport - dt_sub * self.Urhs)
    #         self.Unodes_exp[m+1].assign(self.u_transport)
    def _transport_trajectory(self, n_sub, dtau_list):
        u_idx = self.equation.field_names.index('u')
        for m, dtau_m in enumerate(dtau_list):
            self.u_adv_avg.assign(0.5*(self.Unodes[m].subfunctions[u_idx]+ self.Un.subfunctions[u_idx]))
            dt_sub = float(dtau_m) / n_sub
            self.u_transport.assign(self.Unodes_exp[m])
            for _ in range(n_sub):
                # Stage 1w
                self.Uin.assign(self.u_transport)
                self.solver_rhs_exp.solve()
                self.u_stage.assign(self.u_transport - dt_sub * self.Urhs)
                # Stage 2
                self.Uin.assign(self.u_stage)
                self.solver_rhs_exp.solve()
                self.u_transport.assign(
                    0.5 * self.u_transport + 0.5 * (self.u_stage - dt_sub * self.Urhs)
                )
            self.Unodes_exp[m+1].assign(self.u_transport)
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

        # Add on final implicit terms
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

        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_, old_idx=self.idx),
                                    drop)
        residual += Q
        return residual.form

    @property
    def res_rhs(self):
        """Set up the residual for the calculation of F(y)."""
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
        L = self.residual.label_map(lambda t: any(t.has_label(time_derivative, source_label)),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        L_source = self.residual.label_map(lambda t: t.has_label(source_label),
                                           replace_subject(self.source_in, old_idx=self.idx),
                                           drop)
        residual_rhs = a - (L + L_source)
        return residual_rhs.form

    @property
    def res_rhs_imp(self):
        """Set up the residual for the calculation of F(y)."""
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
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
        a = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Urhs, old_idx=self.idx),
                                    drop)
        residual_adv = self.residual.label_map(
            lambda t: t.has_label(transporting_velocity),
            lambda t: transporting_velocity.update_value(t, self.u_adv_avg)
        )
        L = residual_adv.label_map(lambda t: any(t.has_label(time_derivative, source_label, implicit)),
                                    drop,
                                    replace_subject(self.Uin, old_idx=self.idx))
        L_source = self.residual.label_map(lambda t: t.has_label(source_label),
                                        replace_subject(self.source_in, old_idx=self.idx),
                                        drop)
        residual_rhs = a - (L + L_source)
        return residual_rhs.form
    

    def res(self, m):
        """Set up the discretisation's residual for a given node m."""
        mass_form = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)
        residual = mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.U_DC, old_idx=self.idx))
        residual -= mass_form.label_map(all_terms,
                                        map_if_true=replace_subject(self.U_start, old_idx=self.idx))
        # adv_inc: transport increment from u^n to t^(m), replaces explicit quadrature
        residual -= mass_form.label_map(all_terms,
                                       map_if_true=replace_subject(self.adv_inc, old_idx=self.idx))

        # Add on final implicit terms
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

        Q = self.residual.label_map(lambda t: t.has_label(time_derivative),
                                    replace_subject(self.Q_, old_idx=self.idx),
                                    drop)
        residual += Q
        return residual.form

    @property
    def solver_rhs(self):
        """Set up the problem and the solver for mass matrix inversion."""
        prob_rhs = NonlinearVariationalProblem(self.res_rhs, self.Urhs, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_rhs"
        return NonlinearVariationalSolver(prob_rhs, solver_parameters=self.linear_solver_parameters,
                                          options_prefix=solver_name)

    @cached_property
    def solver_rhs_imp(self):
        """Set up the problem and the solver for implicit mass matrix inversion."""
        prob_rhs = NonlinearVariationalProblem(self.res_rhs_imp, self.Urhs, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_rhs_imp"
        return NonlinearVariationalSolver(prob_rhs, solver_parameters=self.linear_solver_parameters,
                                          options_prefix=solver_name)

    @cached_property
    def solver_rhs_exp(self):
        """Set up the problem and the solver for explicit transport mass matrix inversion."""
        prob_rhs = NonlinearVariationalProblem(self.res_rhs_exp, self.Urhs, bcs=self.bcs)
        solver_name = self.field_name+self.__class__.__name__+"_rhs_exp"
        return NonlinearVariationalSolver(prob_rhs, solver_parameters=self.linear_solver_parameters,
                                          options_prefix=solver_name)

    @cached_property
    def solver(self):
        """Set up a list of solvers for each problem at a node m."""
        m = self.comm.ensemble_comm.rank
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
        #self.u_adv_avg.assign(self.Un.subfunctions[self.equation.field_names.index('u')])
        solver_list = self.solvers
        u_idx = self.equation.field_names.index('u')
        n_sub = self.n_transport_subcycles

        # --------------------------------------------------------------------
        # Predictor: compute initial guess on quadrature nodes
        # --------------------------------------------------------------------
        self.Unodes[0].assign(self.Un)
        self.Unodes_exp[0].assign(self.Un)

        if self.base_flag:
            for m in range(self.M):
                self.base.dt = float(self.dtau[m])
                self.base.apply(self.Unodes[m+1], self.Unodes[m])
        elif self.exp_base is not None:
            # Subcycled forward Euler transport from u^n through all nodes
            self._transport_trajectory(n_sub, self.dtau)
            # Flat initialisation for implicit nodes (transport increment applied in res)
            for m in range(self.M):
                self.Unodes[m+1].assign(self.Un)
            # Diagnostic
            self.adv_inc.assign(self.Unodes_exp[self.M] - self.Un)
            adv_inc_u_norm = self.adv_inc.subfunctions[u_idx].dat.norm
            if self.comm.ensemble_comm.rank == 0:
                logger.info(f"adv_inc u norm start: {adv_inc_u_norm:.6e}")
        else:
            for m in range(self.M):
                self.Unodes[m+1].assign(self.Un)

        for m in range(self.M+1):
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[m], self.base.dt, x_out=self.source_Uk[m])

        # --------------------------------------------------------------------
        # Correction sweeps
        # --------------------------------------------------------------------
        k = 0
        while k < self.maxk:
            k += 1

            if self.qdelta_imp_type == "MIN-SR-FLEX":
                solver = solver_list[k-1]
            else:
                solver = self.solver

            if self.exp_base is not None:
                # Compute implicit tendency at current node for quadrature correction
                self.Uin.assign(self.Unodes[self.comm.ensemble_comm.rank+1])
                self.solver_rhs_imp.solve()
                self.fUnodes[self.comm.ensemble_comm.rank].assign(self.Urhs)

                # Subcycled forward Euler transport from u^n using current (lagged) winds
                # Winds come from Unodes[m] which are updated each sweep via bcast
                self.Unodes_exp[0].assign(self.Un)
                self._transport_trajectory(n_sub, self.dtau)

                # Transport increment at this rank's node
                self.adv_inc.assign(
                    self.Unodes_exp[self.comm.ensemble_comm.rank+1] - self.Un
                )
                # Diagnostic
                adv_inc_u_norm = self.adv_inc.subfunctions[u_idx].dat.norm
                if self.comm.ensemble_comm.rank == 0:
                    logger.info(f"adv_inc u norm: {adv_inc_u_norm:.6e}, sweep k={k}")
            else:
                self.Uin.assign(self.Unodes[self.comm.ensemble_comm.rank+1])
                self.solver_rhs.solve()
                self.fUnodes[self.comm.ensemble_comm.rank].assign(self.Urhs)
                self.adv_inc.assign(Constant(0.0))

            self.compute_quad()

            # Loop through quadrature nodes and solve
            self.Unodes1[0].assign(self.Unodes[0])
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[0], self.base.dt, x_out=self.source_Uk[0])

            # Set Q or S matrix
            self.Q_.assign(self.quad[self.comm.ensemble_comm.rank])

            # Set initial guess for solver
            self.U_DC.assign(self.Unodes[self.comm.ensemble_comm.rank+1])

            # Implicit solve at this node
            solver.solve()
            self.Unodes1[self.comm.ensemble_comm.rank+1].assign(self.U_DC)

            # Evaluate source terms
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes1[self.comm.ensemble_comm.rank+1], self.base.dt, x_out=self.source_Ukp1[self.comm.ensemble_comm.rank+1])

            # Apply limiter if required
            if self.limiter is not None:
                self.limiter.apply(self.Unodes1[self.comm.ensemble_comm.rank+1])

            self.Unodes[self.comm.ensemble_comm.rank+1].assign(self.Unodes1[self.comm.ensemble_comm.rank+1])

            # Share all updated nodal states across ranks for next sweep's transport
            for r in range(self.M):
                self.comm.bcast(self.Unodes[r+1], root=r)

            self.source_Uk[self.comm.ensemble_comm.rank+1].assign(self.source_Ukp1[self.comm.ensemble_comm.rank+1])

        # --------------------------------------------------------------------
        # Final output
        # --------------------------------------------------------------------
        if self.maxk > 0:
            if self.final_update:
                self.Uin.assign(self.Unodes1[self.comm.ensemble_comm.rank+1])
                self.source_in.assign(self.source_Ukp1[self.comm.ensemble_comm.rank+1])
                self.solver_rhs.solve()
                self.fUnodes[self.comm.ensemble_comm.rank].assign(self.Urhs)
                self.compute_quad_final()
                if self.comm.ensemble_comm.rank == self.M-1:
                    self.U_fin.assign(self.Unodes[-1])
                self.comm.bcast(self.U_fin, self.M-1)
                self.solver_fin.solve()
                if self.limiter is not None:
                    self.limiter.apply(self.U_fin)
                x_out.assign(self.U_fin)
            else:
                if self.comm.ensemble_comm.rank == self.M-1:
                    x_out.assign(self.Unodes[-1])
                self.comm.bcast(x_out, self.M-1)
        else:
            if self.comm.ensemble_comm.rank == self.M-1:
                x_out.assign(self.Unodes[-1])
            self.comm.bcast(x_out, self.M-1)
