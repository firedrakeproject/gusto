from rexi import *
from firedrake import Function, TrialFunctions, Constant, \
    LinearVariationalProblem, LinearVariationalSolver, MixedFunctionSpace
from gusto import Configuration, replace_subject, drop, time_derivative


class RexiParameters(Configuration):
    """
    Parameters for the REXI coefficients
    """
    h = 0.2
    M = 64
    reduce_to_half = False


class Rexi(object):
    """
    Class defining the solver for the system

    (A_n + tau L)V_n = U

    required for computing the matrix exponential as described in notes.pdf

    :arg equation: :class:`.Equation` object defining the equation set to
    be solved
    :arg rexi_parameters: :class:`.Equation` object
    :arg solver_parameters: dictionary of solver parameters. Default None,
    which results in the default solver parameters defined in the equation
    class being used.
    :arg manager: :class:`.Ensemble` object containing the space and ensemble
    subcommunicators

    """
    def __init__(self, equation, rexi_parameters, *, solver_parameters=None,
                 manager=None):

        residual = equation.residual

        # Get the Rexi Coefficients, given the values of h and M in
        # rexi_parameters
        self.alpha, self.beta, self.beta2 = RexiCoefficients(rexi_parameters)

        self.manager = manager

        # define the start point of the solver loop (idx) and the
        # number of solvers (N) for this process depending on the
        # total number of solvers (nsolvers) and how many ensemble
        # processes (neprocs) there are.
        nsolvers = len(self.alpha)
        if manager is None:
            # if running in serial we loop over all the solvers, from
            # 0: nsolvers
            self.N = nsolvers
            self.idx = 0
        else:
            rank = manager.ensemble_comm.rank
            neprocs = manager.ensemble_comm.size
            m = int(nsolvers/neprocs)
            p = nsolvers - m*neprocs
            if rank < p:
                self.N = m+1
                self.idx = rank*(m+1)
            else:
                self.N = m
                self.idx = rank*m + p

        # set dummy constants for tau and A_i
        self.ar = Constant(1.)
        self.ai = Constant(1.)
        self.tau = Constant(1.)

        # set up functions, problem and solver
        W_ = equation.function_space
        W = MixedFunctionSpace((W_[0], W_[0], W_[1], W_[1]))
        self.U0 = Function(W)
        ur, ui, hr, hi = self.U0.split()
        U0r = (ur, hr)
        U0i = (ui, hi)
        self.w_sum = Function(W)
        self.w = Function(W)
        self.w_ = Function(W)
        tests = TestFunctions(W)
        trials = TrialFunctions(W)
        tests_r = tests[::2]
        tests_i = tests[1::2]
        trials_r = trials[::2]
        trials_i = trials[1::2]

        ur_mass_form = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(trials_r[0], 0),
            map_if_false=drop)

        ui_mass_form = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(trials_i[0], 0),
            map_if_false=drop)

        hr_mass_form = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(trials_r[1], 1),
            map_if_false=drop)

        hi_mass_form = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(trials_i[1], 1),
            map_if_false=drop)

        Lr = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=drop,
            map_if_false=replace_subject(trials_r))

        Li = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=drop,
            map_if_false=replace_subject(trials_i))

        a = (
            self.ar * ur_mass_form
            + self.tau * Lr
            - self.ai * hr_mass_form
            + self.ar * hi_mass_form
            + self.tau * Li
            + self.ai * ui_mass_form
        )
        L = (
            residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(U0r),
                map_if_false=drop)
            + 
            residual.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_true=replace_subject(U0i),
                map_if_false=drop)
            )

        if hasattr(equation, "aP"):
            aP = equation.aP(trial, self.ai, self.tau)
        else:
            aP = None

        rexi_prob = LinearVariationalProblem(a.form, L.form, self.w, aP=aP,
                                             constant_jacobian=False)

        #if solver_parameters is None:
        #    solver_parameters = equation.solver_parameters

        self.solver = LinearVariationalSolver(
            rexi_prob)#, solver_parameters=solver_parameters)

    def solve(self, U0, dt):
        """
        Solve method for approximating the matrix exponential by a
        rational sum. Solves

        (A_n + tau L)V_n = U

        multiplies by the corresponding B_n and sums over n.

        :arg U0: the mixed function on the rhs.
        :arg dt: the value of tau

        """

        # assign tau and U0 and initialise solution to 0.
        self.tau.assign(dt)
        self.U0.assign(U0)
        self.w_.assign(0.)

        # loop over solvers, assigning a_i, solving and accumulating the sum
        for i in range(self.N):
            j = self.idx + i
            self.ar.assign(self.alpha[j].real)
            self.ai.assign(self.alpha[j].imag)
            self.solver.solve()
            self.w_ += (
                + Constant(self.beta[j].real)*self.w
            )

        # in parallel we have to accumulate the sum over all processes
        if self.manager is not None:
            self.manager.allreduce(self.w_, self.w_sum)
        else:
            self.w_sum.assign(self.w_)

        return self.w_sum
