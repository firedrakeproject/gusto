"""
Objects for discretising time derivatives using time-parallel Deferred Correction
Methods.

This module inherits from the serial SDC and RIDC classes, and implements the
parallelisation of the SDC and RIDC methods using MPI.

SDC parallelises across the quadrature nodes by using diagonal QDelta matrices,
while RIDC parallelises across the correction iterations by using a reduced stencil
and pipelining.
"""

import os
import numpy as np
from firedrake import (
    Function, NonlinearVariationalProblem, NonlinearVariationalSolver, Constant,
    inner, dx, assemble, TestFunction, Cofunction
)
import time
from functools import cached_property
from gusto.time_discretisation.time_discretisation import wrapper_apply
from qmat import genQDeltaCoeffs
from gusto.time_discretisation.deferred_correction import SDC, RIDC
from gusto.core.logging import logger
from gusto.solvers.solver_presets import hybridised_solver_parameters

__all__ = ["Parallel_RIDC", "Parallel_SDC"]


import resource
from mpi4py import MPI
import numpy as np

_node_comm = None


def _owned(f, mode="ro"):
    """Tuple of owned-dof arrays of a Function/Cofunction (one per mixed component)."""
    arrs = f.dat.data_ro if mode == "ro" else f.dat.data_wo
    return arrs if isinstance(arrs, tuple) else (arrs,)


def _flat(f):
    """Owned dofs of all components as one flat float64 array (a copy)."""
    return np.concatenate([a.ravel() for a in _owned(f)])


def _write_flat(f, flat):
    """Write a flat array (layout of _flat) into the owned dofs of f."""
    off = 0
    for a in _owned(f, "wo"):
        a[...] = flat[off:off + a.size].reshape(a.shape)
        off += a.size




def _rss_now_gb():
    """Current resident set size of this process in GB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2  # kB -> GB
    return float("nan")

def _mem(tag):
    """Collective over COMM_WORLD. Logs per-rank peak RSS (max and mean over
    ranks), the largest per-group totals, and the per-group breakdown.
    Groups are nodes by default, or contiguous blocks of MEM_GROUP_SIZE ranks
    if that environment variable is set (= spatial ranks per ensemble member)."""
    import os
    global _node_comm
    comm = MPI.COMM_WORLD
    if _node_comm is None:
        group_size = int(os.environ.get("MEM_GROUP_SIZE", "0"))
        if group_size > 0:
            _node_comm = comm.Split(color=comm.rank // group_size, key=comm.rank)
        else:
            _node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # Linux: kB -> GB
    now = _rss_now_gb()

    peak_max = comm.allreduce(peak, op=MPI.MAX)
    peak_mean = comm.allreduce(peak, op=MPI.SUM) / comm.size
    node_now = _node_comm.allreduce(now, op=MPI.SUM)
    node_peak_sum = _node_comm.allreduce(peak, op=MPI.SUM)
    node_peak_rank_max = _node_comm.allreduce(peak, op=MPI.MAX)
    node_first_rank = _node_comm.allreduce(comm.rank, op=MPI.MIN)
    node_now_max = comm.allreduce(node_now, op=MPI.MAX)
    node_peak_max = comm.allreduce(node_peak_sum, op=MPI.MAX)

    # One entry per group, from each group's leader
    entry = None
    if _node_comm.rank == 0:
        entry = (node_first_rank, MPI.Get_processor_name(), node_now,
                 node_peak_sum, node_peak_rank_max)
    entries = comm.gather(entry, root=0)

    if comm.rank == 0:
        print(f"[mem] {tag:<48s} rank peak max {peak_max:6.3f} GB | "
              f"rank peak mean {peak_mean:6.3f} GB | "
              f"group now {node_now_max:7.1f} GB | "
              f"group sum-of-peaks {node_peak_max:7.1f} GB",
              flush=True)
        for first, host, n_now, n_peak, n_rmax in sorted(e for e in entries if e is not None):
            print(f"[mem]     ranks {first:>4d}+ on {host}: now {n_now:7.1f} GB | "
                  f"sum-of-peaks {n_peak:7.1f} GB | worst rank {n_rmax:6.3f} GB",
                  flush=True)

class Parallel_RIDC(RIDC):
    """Class for Parallel Revisionist Integral Deferred Correction schemes."""

    def __init__(self, base_scheme, domain, M, K, J, output_freq, flush_freq=None, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, communicator=None, imex=True, snes_rtol=1e-5, snes_atol=1e-4):
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
                                            limiter, reduced=True, imex=imex, snes_rtol=snes_rtol, snes_atol=snes_atol)
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
                self.total_ksp_its += self.solver.snes.getLinearSolveIterations()
                self.total_snes_its += self.solver.snes.getIterationNumber()
                self.solver_call_count += 1
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
                self.total_ksp_its += self.solver.snes.getLinearSolveIterations()
                self.total_snes_its += self.solver.snes.getIterationNumber()
                self.solver_call_count += 1
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
                 limiter=None, options=None, initial_guess="base", communicator=None,
                 use_anderson=False, m_aa=2, beta_aa=1.0, aa_start=1, aa_skip_last=False,
                 aa_reg=0.0, aa_kappa=None, aa_fast=True, aa_diag_file=None,
                 imex=True, sweep_tols=None, nonlinear=True):

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
            use_anderson (bool, optional): Whether to accelerate the sweep iteration using
                Anderson mixing (Walker & Ni, 2011, "Type-II"). This is a Qdelta-agnostic,
                opt-in post-processing step applied on top of the unchanged SDC sweep update.
                Defaults to False.
            m_aa (int, optional): Anderson mixing history window size. Should be kept small
                and fixed (2-3 is typical); if `m_aa >= maxk` the history is never evicted
                and the iteration converges to the fully-converged collocation solution,
                independent of qdelta_imp/qdelta_exp. Only used if `use_anderson` is True.
                Defaults to 2.
            beta_aa (float, optional): Anderson mixing damping/relaxation factor in (0, 1].
                A value of 1.0 gives the full (undamped) Anderson step; smaller
                values blend the mixed update back towards the plain sweep output, which
                can improve robustness at the cost of some acceleration. Only used if
                `use_anderson` is True. Defaults to 1.0.
            aa_start (int, optional): first sweep that enters the Anderson history.
            aa_skip_last (bool, optional): return the plain sweep output on the last sweep.
            aa_reg (float, optional): relative Tikhonov regularisation of the Gram matrix.
            aa_kappa (float, optional): step-size safeguard (slow implementation only).
            aa_fast (bool, optional): use the fast numpy implementation of the Anderson
                step. Defaults to True.
            aa_diag_file (str, optional): if given, write per-sweep residual and Anderson
                diagnostics (one CSV row per time step and sweep) to this file from world
                rank 0. Works for plain SDC and for the fast Anderson path. Defaults to None.
        """
        super().__init__(base_scheme, domain, M, maxk, quad_type, node_type, qdelta_imp, qdelta_exp,
                         formulation="Z2N", field_name=field_name,
                         linear_solver_parameters=linear_solver_parameters, nonlinear_solver_parameters=nonlinear_solver_parameters,
                         final_update=final_update,
                         limiter=limiter, initial_guess=initial_guess, imex=imex, sweep_tols=sweep_tols, nonlinear=nonlinear)
        self.comm = communicator
        self.alpha = Constant(0.0)  # Initial value, will be updated in solver setup
        # Checks for parallel SDC
        if self.comm is None:
            raise ValueError("No communicator provided. Please provide a valid MPI communicator.")
        if self.comm.ensemble_comm.size != self.M:
            raise ValueError("Number of ranks must be equal to the number of nodes M for Parallel SDC.")

        # Anderson mixing options
        self.use_anderson = use_anderson
        self.m_aa = m_aa
        self.beta_aa = beta_aa
        self.aa_start = aa_start          # first sweep that enters the AA history
        self.aa_skip_last = aa_skip_last  # return plain G(u) on the final sweep
        self.aa_reg = aa_reg              # relative Tikhonov factor (try 1e-8)
        self.aa_kappa = aa_kappa          # max ||correction|| / ||f_last|| (try 1.0)
        self.aa_fast = aa_fast
        self._MR_hist = []
        self.aa_time = 0.0

        # Per-sweep diagnostics (residual history, Gram conditioning, ...)
        self.aa_diag_file = aa_diag_file
        self._diag = aa_diag_file is not None
        if self._diag and use_anderson and not (aa_fast and aa_kappa is None):
            raise ValueError("aa_diag_file needs aa_fast=True and aa_kappa=None "
                             "when use_anderson is True.")

        if self.use_anderson:
            if self.m_aa < 1:
                raise ValueError("m_aa (Anderson mixing window) must be at least 1.")
            if not (0.0 < self.beta_aa <= 1.0):
                raise ValueError("beta_aa (Anderson mixing damping factor) must be in (0, 1].")
            if self.m_aa >= self.maxk:
                logger.warn("m_aa >= maxk: Anderson mixing history will never be evicted, so "
                            "the SDC iteration will converge towards the fully-converged "
                            "collocation solution, independent of qdelta_imp/qdelta_exp.")
        self._G_hist = []
        self._R_hist = []

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
        _mem("SDC setup: start")
        super(Parallel_SDC, self).setup(equation, apply_bcs, *active_labels)
        _mem("SDC setup: after base setup + SDC fields")
        self._quad_buf = Function(self.W)
        self._mem_step = 0
        if (self.use_anderson and self.aa_fast) or self._diag:
            self._aa_r = Function(self.W)
            self._aa_Mr = Cofunction(self.W.dual())
            self._aa_form = inner(self._aa_r, TestFunction(self.W))*dx
        if self.use_anderson and self.aa_fast:
            self._aa_unew = Function(self.W)
        if self._diag:
            self._diag_setup()

    def compute_quad(self):
        """
        Computes integration of F(y) on quadrature nodes
        """
        rank = self.comm.ensemble_comm.rank
        x = self._quad_buf
        for j in range(self.M):
            x.assign(float(self.Q[j, rank])*self.fUnodes[rank])
            self.comm.reduce(x, self.quad[j], root=j)

    def compute_quad_final(self):
        """
        Computes final integration of F(y) on quadrature nodes
        """
        rank = self.comm.ensemble_comm.rank
        x = self._quad_buf
        x.assign(float(self.Qfin[rank])*self.fUnodes[rank])
        self.comm.allreduce(x, self.quad_final)

    def _anderson_mix(self, Gu, k):
        """
        Anderson-accelerate the SDC sweep iteration (Type-II, Walker & Ni 2011).

        Args:
            Gu (:class:`Function`): raw sweep output G(u) for this rank's node
                (limiter already applied).
            k (int): current sweep number (1-based).

        Returns:
            :class:`Function`: iterate to use as the starting point for the next sweep.
        """
        rank = self.comm.ensemble_comm.rank

        # Plain sweeps before AA starts, and (optionally) on the final sweep.
        # These conditions depend only on k, so every rank takes the same branch.
        if k < self.aa_start or (self.aa_skip_last and k == self.maxk):
            return Gu

        # Residual r = G(u) - u, using the pre-update value of Unodes
        r = Function(self.W)
        r.assign(Gu - self.Unodes[rank+1])

        # Update history, capped at window size m_aa
        self._G_hist.append(Gu.copy(deepcopy=True))
        self._R_hist.append(r.copy(deepcopy=True))
        if len(self._G_hist) > self.m_aa:
            self._G_hist.pop(0)
            self._R_hist.pop(0)

        mm = len(self._R_hist)
        if mm == 1:
            return Gu

        dF = [Function(self.W).assign(self._R_hist[i+1] - self._R_hist[i]) for i in range(mm-1)]
        dG = [Function(self.W).assign(self._G_hist[i+1] - self._G_hist[i]) for i in range(mm-1)]
        f_last = self._R_hist[-1]
        g_last = self._G_hist[-1]

        Gram_local = np.zeros((mm-1, mm-1))
        rhs_local = np.zeros(mm-1)
        for i in range(mm-1):
            rhs_local[i] = assemble(inner(dF[i], f_last)*dx)
            for j in range(mm-1):
                Gram_local[i, j] = assemble(inner(dF[i], dF[j])*dx)

        Gram = self.comm.allreduce(Gram_local)
        rhs = self.comm.allreduce(rhs_local)

        # Relative Tikhonov regularisation (scale-aware, unlike the cond check)
        if self.aa_reg > 0.0:
            Gram = Gram + self.aa_reg*np.trace(Gram)/(mm-1)*np.eye(mm-1)

        try:
            if np.linalg.cond(Gram) > 1e12:
                raise np.linalg.LinAlgError("Anderson Gram matrix is ill-conditioned")
            gamma = self.beta_aa*np.linalg.solve(Gram, rhs)
        except np.linalg.LinAlgError:
            logger.warn("Anderson mixing: Gram matrix singular or ill-conditioned, "
                        "skipping mixing for this sweep.")
            return Gu

        # Extrapolation correction  corr = sum_i gamma_i * dG_i
        corr = Function(self.W)
        for i in range(mm-1):
            corr.assign(corr + Constant(gamma[i])*dG[i])

        # Step-size safeguard: reject if the extrapolation is large compared to
        # the current residual (stacked over all nodes, so all ranks agree).
        if self.aa_kappa is not None:
            local = np.array([assemble(inner(corr, corr)*dx),
                            assemble(inner(f_last, f_last)*dx)])
            tot = self.comm.allreduce(local)
            if tot[0] > self.aa_kappa**2 * tot[1]:
                logger.warn("Anderson mixing: extrapolation step too large, "
                            "skipping mixing for this sweep.")
                return Gu

        u_new = Function(self.W)
        u_new.assign(g_last - corr)
        return u_new

    # ----------------------------------------------------------------------- #
    # Per-sweep diagnostics
    #
    # One CSV row per time step and sweep k (world rank 0 only):
    #   step,k,mm,status,res_total,res_<field>...,cond,gamma_max,pred_ratio
    #
    #   res_total   L2 norm of the stacked residual r = G(U) - U over all nodes and
    #               fields: the residual of the state fed INTO sweep k (the residual
    #               of the very last output is not evaluated)
    #   res_<field> the same per prognostic field (mass matrices are block-diagonal
    #               over the mixed-space components, so these are slices of r . M r)
    #   mm          number of stored sweeps in the Anderson window
    #   status      plain | first | skipped | ok | illcond
    #   cond        condition number of the Gram matrix used in the solve
    #               (trivially 1 for a 1x1 Gram, i.e. m_aa=2)
    #   gamma_max   max |gamma| of the (undamped) least-squares solution
    #   pred_ratio  sqrt(min ||r - dR gamma||^2 / ||r||^2): residual reduction the
    #               Anderson fit predicts (approximate when aa_reg > 0)
    # ----------------------------------------------------------------------- #
    def _diag_setup(self):
        """Slice offsets of the mixed-space components and their field names."""
        sizes = [a.size for a in _owned(self._aa_r)]
        self._diag_off = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        names = list(getattr(self.equation, "field_names", []))
        self._diag_names = (names if len(names) == len(sizes)
                            else [f"c{i}" for i in range(len(sizes))])

    def _diag_res2_local(self, r, Mr):
        """Local squared L2 norm of r per mixed-space component."""
        off = self._diag_off
        return np.array([np.dot(r[off[c]:off[c+1]], Mr[off[c]:off[c+1]])
                         for c in range(len(off) - 1)])

    def _diag_reduce(self, r, Mr):
        """Global (all nodes, all spatial ranks) squared residual norm per component."""
        v = self._diag_res2_local(r, Mr)
        self.comm.global_comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
        return v

    def _diag_record(self, k, res2, mm, status, cond=np.nan, gamma_max=np.nan, pred=np.nan):
        """Append one row to the diagnostics CSV (world rank 0 only)."""
        if self.comm.global_comm.rank != 0:
            return
        new_file = (not os.path.exists(self.aa_diag_file)
                    or os.path.getsize(self.aa_diag_file) == 0)
        total = float(np.sum(res2))
        with open(self.aa_diag_file, "a") as f:
            if new_file:
                f.write("step,k,mm,status,res_total,"
                        + ",".join(f"res_{n}" for n in self._diag_names)
                        + ",cond,gamma_max,pred_ratio\n")
            f.write(f"{self._mem_step},{k},{mm},{status},{np.sqrt(total):.6e},"
                    + ",".join(f"{np.sqrt(max(float(v), 0.0)):.6e}" for v in res2)
                    + f",{cond:.6e},{gamma_max:.6e},{pred:.6e}\n")

    def _diag_plain(self, k, Gu):
        """Residual diagnostics for a plain (non-Anderson) sweep."""
        rank = self.comm.ensemble_comm.rank
        self._aa_r.assign(Gu - self.Unodes[rank+1])
        assemble(self._aa_form, tensor=self._aa_Mr)
        res2 = self._diag_reduce(_flat(self._aa_r), _flat(self._aa_Mr))
        self._diag_record(k, res2, 0, "plain")

    def _anderson_mix_fast(self, Gu, k):
        """
        Same algorithm as _anderson_mix (Type-II, L2 inner product, aa_start /
        aa_skip_last / aa_reg / beta_aa supported) using numpy on owned dofs,
        with optional per-sweep diagnostics (see above).

        Args:
            Gu (:class:`Function`): raw sweep output G(u) for this rank's node.
            k (int): current sweep number (1-based).

        Returns:
            :class:`Function`: iterate to use as the start of the next sweep.
        """
        rank = self.comm.ensemble_comm.rank

        # Decided from k only, so every rank takes the same branch
        skip = k < self.aa_start or (self.aa_skip_last and k == self.maxk)
        if skip and not self._diag:
            return Gu

        # r = G(u) - u and its mass-weighted counterpart M r (one assemble)
        self._aa_r.assign(Gu - self.Unodes[rank+1])
        assemble(self._aa_form, tensor=self._aa_Mr)
        r = _flat(self._aa_r)
        Mr = _flat(self._aa_Mr)

        if skip:
            self._diag_record(k, self._diag_reduce(r, Mr), 0, "skipped")
            return Gu

        self._G_hist.append(_flat(Gu))
        self._R_hist.append(r)
        self._MR_hist.append(Mr)
        if len(self._G_hist) > self.m_aa:
            self._G_hist.pop(0)
            self._R_hist.pop(0)
            self._MR_hist.pop(0)

        mm = len(self._R_hist)
        if mm == 1:
            if self._diag:
                self._diag_record(k, self._diag_reduce(r, Mr), mm, "first")
            return Gu

        G = np.asarray(self._G_hist)       # (mm, n)
        R = np.asarray(self._R_hist)
        MR = np.asarray(self._MR_hist)
        dF = R[1:] - R[:-1]                # (mm-1, n)
        MdF = MR[1:] - MR[:-1]

        nn = (mm-1)*(mm-1)
        nd = (len(self._diag_off) - 1) if self._diag else 0
        local = np.empty(nn + (mm-1) + nd)
        local[:nn] = (dF @ MdF.T).ravel()      # local part of dF^T M dF
        local[nn:nn+mm-1] = dF @ MR[-1]        # local part of dF^T M f_last
        if self._diag:
            local[nn+mm-1:] = self._diag_res2_local(r, Mr)

        # One reduction: spatial ranks x ensemble members == all world ranks
        self.comm.global_comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)
        Gram = local[:nn].reshape(mm-1, mm-1)
        Gram = 0.5*(Gram + Gram.T)
        rhs = local[nn:nn+mm-1].copy()
        res2 = local[nn+mm-1:].copy() if self._diag else None

        if self.aa_reg > 0.0:
            Gram = Gram + self.aa_reg*np.trace(Gram)/(mm-1)*np.eye(mm-1)

        cond = np.linalg.cond(Gram)
        # Every rank holds identical Gram/rhs, so every rank branches identically
        try:
            if cond > 1e12:
                raise np.linalg.LinAlgError("Anderson Gram matrix is ill-conditioned")
            gamma0 = np.linalg.solve(Gram, rhs)
        except np.linalg.LinAlgError:
            logger.warn("Anderson mixing: Gram matrix singular or ill-conditioned, "
                        "skipping mixing for this sweep.")
            if self._diag:
                self._diag_record(k, res2, mm, "illcond", cond=cond)
            return Gu
        gamma = self.beta_aa*gamma0

        if self._diag:
            total = float(res2.sum())
            pred = (np.sqrt(max(total - float(rhs @ gamma0), 0.0)/total)
                    if total > 0.0 else np.nan)
            self._diag_record(k, res2, mm, "ok", cond=cond,
                              gamma_max=float(np.max(np.abs(gamma0))), pred=pred)

        # u_new = G_last - sum_i gamma_i (G_{i+1} - G_i) = sum_k c_k G_k
        c = np.zeros(mm)
        c[-1] = 1.0
        for i, g in enumerate(gamma):
            c[i+1] -= g
            c[i] += g
        _write_flat(self._aa_unew, c @ G)
        return self._aa_unew

    @cached_property
    def solver(self):
        """Set up the solver for this rank's quadrature node."""
        import os
        m = self.comm.ensemble_comm.rank
        # setup solver using residual defined in derived class
        self.alpha.assign(self.Qdelta_imp[m, m]/float(self.dt_coarse))
        QD_imp = genQDeltaCoeffs(
                    self.qdelta_imp_type,
                    form=self.formulation,
                    nodes=self.nodes/float(self.dt_coarse),
                    Q=self.Q/float(self.dt_coarse),
                    nNodes=self.M,
                    nodeType=self.node_type,
                    quadType=self.quad_type
                )
        alpha = QD_imp[m, m]
        # Diagnostic: build the hybridised PC with another member's exact alpha
        # (residual unchanged; exact value, so no new kernels are generated)
        force_member = os.environ.get("SDC_FORCE_MEMBER")
        if force_member is not None:
            alpha = QD_imp[int(force_member), int(force_member)]
        if self.comm.comm.rank == 0:
            print(f"[alpha] ensemble member {m}: PC alpha = {alpha!r}", flush=True)
        if self.nonlinear_solver_parameters is None:
            self.nonlinear_solver_parameters, self.appctx = hybridised_solver_parameters(self.equation, self.equation.field_names, alpha=alpha, tau_values=None, nonlinear=self.nonlinear, imex=self.imex)
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
        for k in range(1, self.maxk+1):
            # Recompute Implicit Q_delta matrix for each iteration k
            Qdelta_imp_k = float(self.dt_coarse)*genQDeltaCoeffs(
                self.qdelta_imp_type,
                form=self.formulation,
                nodes=self.nodes,
                Q=self.Q,
                nNodes=self.M,
                nodeType=self.node_type,
                quadType=self.quad_type,
                k=k
            )
            alpha_k = Qdelta_imp_k[self.comm.ensemble_comm.rank, self.comm.ensemble_comm.rank]/float(self.dt_coarse)
            if self.nonlinear_solver_parameters is not None:
                nonlinear_solver_parameters_k, appctx_k = hybridised_solver_parameters(self.equation, self.equation.field_names, alpha=alpha_k, tau_values=None, nonlinear=self.nonlinear, imex=self.imex)
            else:
                nonlinear_solver_parameters_k = None
                appctx_k = None
            problem_k = NonlinearVariationalProblem(self.res(self.comm.ensemble_comm.rank), self.U_DC, bcs=self.bcs)
            solver_name_k = self.field_name+self.__class__.__name__ + "%s_k%s" % (self.comm.ensemble_comm.rank, k)
            solver_k = NonlinearVariationalSolver(problem_k, solver_parameters=nonlinear_solver_parameters_k, appctx=appctx_k, options_prefix=solver_name_k)
            solvers.append(solver_k)
        return solvers

    @wrapper_apply
    def apply(self, x_out, x_in):
        self._mem_step = getattr(self, "_mem_step", 0) + 1
        first = self._mem_step == 1

        self.Un.assign(x_in)
        self.U_start.assign(self.Un)
        # Per-sweep solvers are only needed for MIN-SR-FLEX
        solver_list = self.solvers if self.qdelta_imp_type == "MIN-SR-FLEX" else None

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
        if first:
            _mem("step 1: after initial guess")

        # Reset Anderson mixing history for this time step
        if self.use_anderson:
            self._G_hist = []
            self._R_hist = []
            self._MR_hist = []

        # Iterate through correction sweeps
        k = 0
        while k < self.maxk:
            k += 1

            if self.qdelta_imp_type == "MIN-SR-FLEX":
                # Recompute Implicit Q_delta matrix for each iteration k
                self.Qdelta_imp = float(self.dt_coarse)*genQDeltaCoeffs(
                    self.qdelta_imp_type,
                    form=self.formulation,
                    nodes=self.nodes,
                    Q=self.Q,
                    nNodes=self.M,
                    nodeType=self.node_type,
                    quadType=self.quad_type,
                    k=k
                )
                self.alpha.assign(self.Qdelta_imp[self.comm.ensemble_comm.rank, self.comm.ensemble_comm.rank]/float(self.dt_coarse))
                solver = solver_list[k-1]

            # Z2N: sum(j=1,M) (q_mj*F(y_m^k) + q_mj*S(y_m^k))
            self.Uin.assign(self.Unodes[self.comm.ensemble_comm.rank+1])
            for evaluate in self.evaluate_source:
                evaluate(self.Uin, self.base.dt, x_out=self.source_in)
            self.solver_rhs.solve()
            self.fUnodes[self.comm.ensemble_comm.rank].assign(self.Urhs)
            if first and k == 1:
                _mem("step 1 k=1: after first solver_rhs (mass aij+ILU)")

            self.compute_quad()
            if first and k == 1:
                _mem("step 1 k=1: after compute_quad")

            self.Unodes1[0].assign(self.Unodes[0])
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes[0], self.base.dt, x_out=self.source_Uk[0])

            # Set Q matrix contribution for this node
            self.Q_.assign(self.quad[self.comm.ensemble_comm.rank])

            # Set initial guess for solver
            self.U_DC.assign(self.Unodes[self.comm.ensemble_comm.rank+1])
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

            self.solver.solve()
            self.total_snes_its += self.solver.snes.getIterationNumber()
            self.total_ksp_its += self.solver.snes.getLinearSolveIterations()
            self.solver_call_count += 1
            self.Unodes1[self.comm.ensemble_comm.rank+1].assign(self.U_DC)
            if first and k <= 2:
                _mem(f"step 1 k={k}: after hybridised solve")

            # Evaluate source terms
            for evaluate in self.evaluate_source:
                evaluate(self.Unodes1[self.comm.ensemble_comm.rank+1], self.base.dt, x_out=self.source_Ukp1[self.comm.ensemble_comm.rank+1])

            # Apply limiter if required
            if self.limiter is not None:
                self.limiter.apply(self.Unodes1[self.comm.ensemble_comm.rank+1])

            if self.use_anderson:
                t0 = time.perf_counter()
                mixer = (self._anderson_mix_fast
                            if (self.aa_fast and self.aa_kappa is None)
                            else self._anderson_mix)
                u_new = mixer(self.Unodes1[self.comm.ensemble_comm.rank+1], k)
                self.aa_time += time.perf_counter() - t0
                self.Unodes[self.comm.ensemble_comm.rank+1].assign(u_new)
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes[self.comm.ensemble_comm.rank+1])
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes[self.comm.ensemble_comm.rank+1], self.base.dt,
                             x_out=self.source_Uk[self.comm.ensemble_comm.rank+1])
            else:
                if self._diag:
                    self._diag_plain(k, self.Unodes1[self.comm.ensemble_comm.rank+1])
                self.Unodes[self.comm.ensemble_comm.rank+1].assign(self.Unodes1[self.comm.ensemble_comm.rank+1])
                self.source_Uk[self.comm.ensemble_comm.rank+1].assign(self.source_Ukp1[self.comm.ensemble_comm.rank+1])

        if self.maxk > 0:
            if self.final_update:
                self.Uin.assign(self.Unodes[self.comm.ensemble_comm.rank+1])
                self.source_in.assign(self.source_Uk[self.comm.ensemble_comm.rank+1])
                self.solver_rhs.solve()
                self.fUnodes[self.comm.ensemble_comm.rank].assign(self.Urhs)
                self.compute_quad_final()
                # y_(n+1) = y_n + sum(j=1,M) q_j*F(y_j)
                if self.comm.ensemble_comm.rank == self.M-1:
                    self.U_fin.assign(self.Unodes[-1])
                self.comm.bcast(self.U_fin, self.M-1)
                self.solver_fin.solve()
                if first:
                    _mem("step 1: after first solver_fin (mass aij+ILU)")
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

        if self._mem_step <= 2 or self._mem_step % 12 == 0:
            _mem(f"end of step {self._mem_step}")