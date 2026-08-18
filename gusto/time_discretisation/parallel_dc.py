"""
Objects for discretising time derivatives using time-parallel Deferred Correction
Methods.

This module inherits from the serial SDC and RIDC classes, and implements the
parallelisation of the SDC and RIDC methods using MPI.

SDC parallelises across the quadrature nodes by using diagonal QDelta matrices,
while RIDC parallelises across the correction iterations by using a reduced stencil
and pipelining.
"""

from firedrake import Function, NonlinearVariationalProblem, NonlinearVariationalSolver
from firedrake.assemble import get_assembler
from firedrake.fml import replace_subject, drop
from gusto.core.labels import time_derivative, source_label
from functools import cached_property
from gusto.time_discretisation.time_discretisation import wrapper_apply
from gusto.time_discretisation.deferred_correction import SDC, RIDC
from gusto.core.logging import logger
from mpi4py import MPI

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
        # self.TAG_EXCHANGE_FIELD = 11  # Tag for sending nodal fields (Firedrake Functions)
        # self.TAG_EXCHANGE_SOURCE = self.TAG_EXCHANGE_FIELD + J  # Tag for sending nodal source fields (Firedrake Functions)
        # self.TAG_FLUSH_PIPE = self.TAG_EXCHANGE_SOURCE + J  # Tag for flushing pipe and restarting
        # self.TAG_FINAL_OUT = self.TAG_FLUSH_PIPE + J  # Tag for the final broadcast and output
        # self.TAG_END_INTERVAL = self.TAG_FINAL_OUT + J  # Tag for telling the rank above you that you have ended interval j

            
        self.TAG_STEP_MOD = max(32, 8 * (self.K + 1))
        self.TAG_NODE_STRIDE = self.TAG_STEP_MOD
        self.TAG_CHANNEL_STRIDE = self.TAG_NODE_STRIDE * (self.M + 2)

        self.TAG_CHANNEL_FIELD = 0
        self.TAG_CHANNEL_SOURCE = 1
        self.TAG_CHANNEL_FLUSH = 2
        self.TAG_CHANNEL_FINAL = 3
        self.TAG_CHANNEL_END_INTERVAL = 4
        self._n_tag_channels = 5

        # Sanity-check against the MPI implementation's tag upper bound.
        max_tag = (self._n_tag_channels - 1) * self.TAG_CHANNEL_STRIDE \
            + (self.M + 1) * self.TAG_NODE_STRIDE + (self.TAG_STEP_MOD - 1)
        tag_ub = self.comm.ensemble_comm.Get_attr(MPI.TAG_UB)# MPI.TAG_UB has value 3 as a predefined keyval
        if tag_ub is not None and max_tag >= tag_ub:
            raise ValueError(
                f"Constructed tag range (max={max_tag}) exceeds MPI_TAG_UB "
                f"({tag_ub}) for this M, K. Increase margin or reduce M/K."
            )

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
    
    def _tag(self, channel, m=0, step=None):
        """Build a collision-free MPI tag for a given channel and subinterval."""
        step = self.step if step is None else step
        return (channel * self.TAG_CHANNEL_STRIDE
                + m * self.TAG_NODE_STRIDE
                + (step % self.TAG_STEP_MOD))

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
        x_out.assign(x_in)
        self.kval = self.comm.ensemble_comm.rank
        self.Un.assign(x_in)
        self.Unodes[0].assign(self.Un)

        if (self.flush_freq > 0 and (self.step - 1) % self.flush_freq == 0):
            self.Unodes[0].assign(x_in)
        else:
            self.Unodes[0].assign(self.Uprev)
        self.Unodes1[0].assign(x_in)
        for evaluate in self.evaluate_source:
            evaluate(self.Unodes[0], self.dt_coarse, x_out=self.source_Uk[0])

        if (self.comm.ensemble_comm.rank == 0):
            for m in range(self.M):
                self.base.dt = float(self.dt)
                self.base.apply(self.Unodes[m + 1], self.Unodes[m])
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes[m + 1], self.dt_coarse, x_out=self.source_Uk[m + 1])

                self.U_send[m + 1].assign(self.Unodes[m + 1])
                self.comm.isend(self.U_send[m + 1], dest=self.kval + 1,
                                tag=self._tag(self.TAG_CHANNEL_FIELD, m + 1))
        else:
            for m in range(1, self.kval + 1):
                self.comm.recv(self.U_send[m], source=self.kval - 1,
                                tag=self._tag(self.TAG_CHANNEL_FIELD, m))
                self.Unodes[m].assign(self.U_send[m])
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes[m], self.dt_coarse, x_out=self.source_Uk[m])

            for m in range(0, self.kval):
                self.rhs_assemblers[self.kval - 1][m].assemble(tensor=self.b)

                self.U_start.assign(self.Unodes1[m])
                self.Ukp1_m.assign(self.Unodes1[m])
                self.Uk_mp1.assign(self.Unodes[m + 1])
                self.Uk_m.assign(self.Unodes[m])
                self.source_Ukp1_m.assign(self.source_Ukp1[m])
                self.source_Uk_m.assign(self.source_Uk[m])
                self.U_DC.assign(self.Unodes[m + 1])

                self._lag_reset_solver()
                self.solver.solve()
                # Update iteration counters
                self.total_ksp_its += self.solver.snes.getLinearSolveIterations()
                self.total_snes_its += self.solver.snes.getIterationNumber()

                self._lag_note_solver_call()
                self.Unodes1[m + 1].assign(self.U_DC)

                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes1[m + 1], self.dt_coarse, x_out=self.source_Ukp1[m + 1])

                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m + 1])

                if self.kval < self.K:
                    self.U_send[m + 1].assign(self.Unodes1[m + 1])
                    self.comm.isend(self.U_send[m + 1], dest=self.kval + 1,
                                    tag=self._tag(self.TAG_CHANNEL_FIELD, m + 1))

            for m in range(self.kval, self.M):
                self.comm.recv(self.U_send[m + 1], source=self.kval - 1,
                                tag=self._tag(self.TAG_CHANNEL_FIELD, m + 1))
                self.Unodes[m + 1].assign(self.U_send[m + 1])
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes[m + 1], self.dt_coarse, x_out=self.source_Uk[m + 1])

                self.rhs_assemblers[self.kval - 1][m].assemble(tensor=self.b)

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

                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m + 1])

                if self.kval < self.K:
                    self.U_send[m + 1].assign(self.Unodes1[m + 1])
                    self.comm.isend(self.U_send[m + 1], dest=self.kval + 1,
                                    tag=self._tag(self.TAG_CHANNEL_FIELD, m + 1))

        if (self.flush_freq > 0 and self.step % self.flush_freq == 0) or self.step == self.J:
            if (self.kval == self.K):
                x_out.assign(self.Unodes1[-1])
                for i in range(self.K):
                    self.comm.isend(x_out, dest=i, tag=self._tag(self.TAG_CHANNEL_FLUSH))
            else:
                self.comm.recv(x_out, source=self.K, tag=self._tag(self.TAG_CHANNEL_FLUSH))
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
                 limiter=None, options=None, initial_guess="base", communicator=None, sweep_tols=None):
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
            sweep_tols (list of dict, optional): list of tolerances for each sweep. Defaults to None.
        """
        super().__init__(base_scheme, domain, M, maxk, quad_type, node_type, qdelta_imp, qdelta_exp,
                         formulation="Z2N", field_name=field_name,
                         linear_solver_parameters=linear_solver_parameters, nonlinear_solver_parameters=nonlinear_solver_parameters,
                         final_update=final_update,
                         limiter=limiter, initial_guess=initial_guess, sweep_tols=sweep_tols)
        self.comm = communicator
        # Checks for parallel SDC
        if self.comm is None:
            raise ValueError("No communicator provided. Please provide a valid MPI communicator.")
        if self.comm.ensemble_comm.size != self.M:
            raise ValueError("Number of ranks must be equal to the number of nodes M for Parallel SDC.")
        
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

        _ = self.solvers

        # Rank-local storage for parallel quadrature via MPI reduce/allreduce.
        self.fUnodes = [Function(self.W) for _ in range(self.M)]
        self.quad = [Function(self.W) for _ in range(self.M)]
        self.quad_final = Function(self.W)
        self.Urhs = Function(self.W)
        self.Uin = Function(self.W)
        self.source_in = Function(self.W)
    
    @cached_property
    def Qf_assembler(self):
        """Cached assembler for this rank's own Qf[m], m = rank's owned node."""
        m = self.comm.ensemble_comm.rank
        return get_assembler(self.Qf_form(m), tensor=self.Qf[m])

    def _exchange_nodes(self, node_list):
        """
        Make every rank's copy of node_list up to date for all M nodes.
        """
        for i in range(self.M):
            self.comm.bcast(node_list[i + 1], root=i)
    
    @cached_property
    def solvers(self):
        """Build rank-local SDC solvers: one per rank, or one per sweep for MIN-SR-FLEX."""
        m = self.comm.ensemble_comm.rank
        if self.qdelta_imp_type == "MIN-SR-FLEX":
            return [self._build_solver(m, k) for k in range(1, self.maxk + 1)]
        return self._build_solver(m)
    @wrapper_apply
    def apply(self, x_out, x_in):
        self.Un.assign(x_in)
        self.U_start.assign(self.Un)
        solver_list = self.solvers
        self._lag_reset([solver_list])

        rank = self.comm.ensemble_comm.rank

        # Initial guess: every rank runs the same serial base-scheme sweep
        # redundantly, so all M+1 entries are already consistent -- no
        # exchange needed before the first sweep's Qf assembly.
        self.Unodes[0].assign(self.Un)
        if self.base_flag:
            for m in range(self.M):
                self.base.dt = float(self.dtau[m])
                self.base.apply(self.Unodes[m+1], self.Unodes[m])
        else:
            for m in range(self.M):
                self.Unodes[m+1].assign(self.Un)
        for m in range(self.M+1):
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[m], self.base.dt, x_out=self.source_Uk[m])

        k = 0
        while k < self.maxk:
            k += 1
            solver = solver_list[k-1] if self.qdelta_imp_type == "MIN-SR-FLEX" else solver_list

            # Direct weak-form Qf
            self.Qf_assembler.assemble(tensor=self.Qf[rank])

            self.Unodes1[0].assign(self.Unodes[0])
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[0], self.base.dt, x_out=self.source_Uk[0])

            self.U_DC.assign(self.Unodes[rank+1])

            if self.sweep_tols is not None:
                tol = self.sweep_tols[k-1]
                solver.snes.ksp.setTolerances(atol=tol["ksp_atol"], rtol=tol["ksp_rtol"])
                solver.snes.setTolerances(atol=tol["snes_atol"], rtol=tol["snes_rtol"])

            solver.solve()
            # Update iteration counters
            self.total_ksp_its += solver.snes.getLinearSolveIterations()
            self.total_snes_its += solver.snes.getIterationNumber()

            self.Unodes1[rank+1].assign(self.U_DC)

            for evaluate in self.evaluate_source:
                evaluate(self.Unodes1[rank+1], self.base.dt, x_out=self.source_Ukp1[rank+1])

            if self.limiter is not None:
                self.limiter.apply(self.Unodes1[rank+1])

            # Commit this rank's own update locally
            self.Unodes[rank+1].assign(self.Unodes1[rank+1])
            self.source_Uk[rank+1].assign(self.source_Ukp1[rank+1])

            # Exchange so every rank's y^k is current for the next
            # sweep's Qf (or, on the last sweep, for the final update).
            self._exchange_nodes(self.Unodes)
            self._exchange_nodes(self.source_Uk)
            for m in range(1, self.M+1):
                self.Unodes1[m].assign(self.Unodes[m])

        if self.maxk > 0:
            if self.final_update:
                self.compute_Qf_fin()
                self.U_fin.assign(self.Unodes[-1])
                self.solver_fin.solve()
                if self.limiter is not None:
                    self.limiter.apply(self.U_fin)
                x_out.assign(self.U_fin)
            else:
                x_out.assign(self.Unodes[-1])
        else:
            x_out.assign(self.Unodes[-1])

        self._step_count = self._step_count + 1
