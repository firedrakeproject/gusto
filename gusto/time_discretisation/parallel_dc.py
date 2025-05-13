u"""
Objects for discretising time derivatives using time-parallel Deferred Correction
Methods.

This module inherits from the serial SDC and RIDC classes, and implements the
parallelisation of the SDC and RIDC methods using MPI.

SDC parallelises across the quadrature nodes by using diagonal QDelta matrices,
while RIDC parallelises across the correction iterations by using a reduced stencil
and pipelining.
"""

from firedrake import (
    Function
)
from gusto.time_discretisation.time_discretisation import wrapper_apply
from qmat import genQDeltaCoeffs
from gusto.time_discretisation.deferred_correction import SDC, RIDC

__all__ = ["Parallel_RIDC", "Parallel_SDC"]


class Parallel_RIDC(RIDC):
    """Class for Parallel Revisionist Integral Deferred Correction schemes."""

    def __init__(self, base_scheme, domain, M, K, field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None,
                 limiter=None, options=None, communicator=None):
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
            communicator (MPI communicator, optional): communicator for parallel execution. Defaults to None.
        """

        super(Parallel_RIDC, self).__init__(base_scheme, domain, M, K, field_name,
                                            linear_solver_parameters, nonlinear_solver_parameters,
                                            limiter, options, reduced=True)
        self.comm = communicator

        # Checks for parallel RIDC
        if self.comm is None:
            raise ValueError("No communicator provided. Please provide a valid MPI communicator.")
        if self.comm.ensemble_comm.size != self.K + 1:
            raise ValueError("Number of ranks must be equal to K+1 for Parallel RIDC.")
        if self.M < self.K*(self.K+1)//2:
            raise ValueError("Number of subintervals M must be greater than K*(K+1)/2 for Parallel RIDC.")

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
        super(Parallel_RIDC, self).setup(equation, apply_bcs, *active_labels)

        self.Uk_mp1 = Function(self.W)
        self.Uk_m = Function(self.W)
        self.Ukp1_m = Function(self.W)

    @wrapper_apply
    def apply(self, x_out, x_in):
        # Set up varibles on this rank
        x_out.assign(x_in)
        self.kval = self.comm.ensemble_comm.rank
        self.Un.assign(x_in)
        self.Unodes[0].assign(self.Un)
        # Loop through quadrature nodes and solve
        self.Unodes1[0].assign(self.Unodes[0])
        for evaluate in self.evaluate_source:
            evaluate(self.Unodes[0], self.base.dt, x_out=self.source_Uk[0])
        self.Uin.assign(self.Unodes[0])
        self.solver_rhs.solve()
        self.fUnodes[0].assign(self.Urhs)

        # On first communicator, we do the predictor step
        if (self.comm.ensemble_comm.rank == 0):
            # Base timestepper
            for m in range(self.M):
                self.base.dt = float(self.dt)
                self.base.apply(self.Unodes[m+1], self.Unodes[m])
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes[m+1], self.base.dt, x_out=self.source_Uk[m+1])

                # Send base guess to k+1 correction
                self.comm.send(self.Unodes[m+1], dest=self.kval+1, tag=100+m+1)
        else:
            for m in range(1, self.kval + 1):
                # Recieve and evaluate the stencil of guesses we need to correct
                self.comm.recv(self.Unodes[m], source=self.kval-1, tag=100+m)
                self.Uin.assign(self.Unodes[m])
                for evaluate in self.evaluate_source:
                    evaluate(self.Uin, self.base.dt, x_out=self.source_in)
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
                    self.comm.send(self.Unodes1[m+1], dest=self.kval+1, tag=100+m+1)

            for m in range(self.kval, self.M):
                # Recieve the guess we need to correct and evaluate the rhs
                self.comm.recv(self.Unodes[m+1], source=self.kval-1, tag=100+m+1)
                self.Uin.assign(self.Unodes[m+1])
                for evaluate in self.evaluate_source:
                    evaluate(self.Uin, self.base.dt, x_out=self.source_in)
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

                # Evaluate source terms
                for evaluate in self.evaluate_source:
                    evaluate(self.Unodes1[m+1], self.base.dt, x_out=self.source_Ukp1[m+1])

                # Apply limiter if required
                if self.limiter is not None:
                    self.limiter.apply(self.Unodes1[m+1])

                # Send our updated value to next communicator
                if self.kval < self.K:
                    self.comm.send(self.Unodes1[m+1], dest=self.kval+1, tag=100+m+1)

        if (self.kval == self.K):
            # Broadcast the final result to all other ranks
            x_out.assign(self.Unodes1[-1])
            for i in range(self.K):
                # Send the final result to all other ranks
                self.comm.send(x_out, dest=i, tag=200)
        else:
            # Receive the final result from rank K
            self.comm.recv(x_out, source=self.K, tag=200)


class Parallel_SDC(SDC):
    """Class for Spectral Deferred Correction schemes."""

    def __init__(self, base_scheme, domain, M, maxk, quad_type, node_type, qdelta_imp, qdelta_exp,
                 field_name=None,
                 linear_solver_parameters=None, nonlinear_solver_parameters=None, final_update=True,
                 limiter=None, options=None, initial_guess="base", communicator=None):
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
            communicator (MPI communicator, optional): communicator for parallel execution. Defaults to None.
        """
        super().__init__(base_scheme, domain, M, maxk, quad_type, node_type, qdelta_imp, qdelta_exp,
                         formulation="Z2N", field_name=field_name,
                         linear_solver_parameters=linear_solver_parameters, nonlinear_solver_parameters=nonlinear_solver_parameters,
                         final_update=final_update,
                         limiter=limiter, options=options, initial_guess=initial_guess)
        self.comm = communicator

        # Checks for parallel SDC
        if self.comm is None:
            raise ValueError("No communicator provided. Please provide a valid MPI communicator.")
        if self.comm.ensemble_comm.size != self.M:
            raise ValueError("Number of ranks must be equal to the number of nodes M for Parallel SDC.")

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
            self.Uin.assign(self.Unodes[self.comm.ensemble_comm.rank+1])
            # Include source terms
            for evaluate in self.evaluate_source:
                evaluate(self.Uin, self.base.dt, x_out=self.source_in)
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
            self.solver = solver_list[self.comm.ensemble_comm.rank]
            self.U_DC.assign(self.Unodes[self.comm.ensemble_comm.rank+1])

            # Compute
            # for N2N:
            # y_m^(k+1) = y_(m-1)^(k+1) + dtau_m*(F(y_(m)^(k+1)) - F(y_(m)^k)
            #             + S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
            #             + sum(j=1,M) s_mj*(F+S)(y^k)
            # for Z2N:
            # y_m^(k+1) = y^n + sum(j=1,m) Qdelta_imp[m,j]*(F(y_(m)^(k+1)) - F(y_(m)^k))
            #             + sum(j=1,M)  Q_delta_exp[m,j]*(S(y_(m-1)^(k+1)) - S(y_(m-1)^k))
            self.solver.solve()
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
            x_out.assign(self.Unodes[-1])
