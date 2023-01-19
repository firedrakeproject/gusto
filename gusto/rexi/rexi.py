from rexi import *
from firedrake import Function, TrialFunctions, Constant, \
    LinearVariationalProblem, LinearVariationalSolver, MixedFunctionSpace
from gusto import Configuration, replace_subject, drop, time_derivative, all_terms, replace_test_function, prognostic, Term
from firedrake.formmanipulation import split_form


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
        self.w_out = Function(W_)
        W = MixedFunctionSpace((W_[0], W_[0], W_[1], W_[1]))
        self.U0 = Function(W)
        ur, ui, hr, hi = self.U0.split()
        self.w_sum = Function(W)
        self.w = Function(W)
        self.w_ = Function(W)
        tests = TestFunctions(W)
        trials = TrialFunctions(W)
        tests_r = tests[::2]
        tests_i = tests[1::2]
        trials_r = trials[::2]
        trials_i = trials[1::2]

        mass_form = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)
        u_mass_form = mass_form.label_map(
            lambda t: t.get(prognostic) == "u",
            lambda t: Term(
                split_form(t.form)[0].form,
                t.labels),
            drop)
        h_mass_form = mass_form.label_map(
            lambda t: t.get(prognostic) == "D",
            lambda t: Term(
                split_form(t.form)[1].form,
                t.labels),
            drop)
        
        ur_mass_form = u_mass_form.label_map(
            all_terms,
            replace_test_function(tests_r[0]))
        aur_mass_form = ur_mass_form.label_map(
            all_terms,
            replace_subject(trials_r[0], 0))

        hr_mass_form = h_mass_form.label_map(
            all_terms,
            replace_test_function(tests_r[1]))
        ahr_mass_form = hr_mass_form.label_map(
            all_terms,
            replace_subject(trials_r[1], 1))

        ui_mass_form = u_mass_form.label_map(
            all_terms,
            replace_test_function(tests_i[0]))
        aui_mass_form = ui_mass_form.label_map(
            all_terms,
            replace_subject(trials_i[0], 0))

        hi_mass_form = h_mass_form.label_map(
            all_terms,
            replace_test_function(tests_i[1]))
        ahi_mass_form = hi_mass_form.label_map(
            all_terms,
            replace_subject(trials_i[1], 1))

        L_form = residual.label_map(
            lambda t: t.has_label(time_derivative),
            drop)

        Lu_form = L_form.label_map(
            lambda t: t.get(prognostic) == "u",
            lambda t: Term(
                split_form(t.form)[0].form,
                t.labels),
            drop)

        Lh_form = L_form.label_map(
            lambda t: t.get(prognostic) == "D",
            lambda t: Term(
                split_form(t.form)[1].form,
                t.labels),
            drop)
        
        Lru = Lu_form.label_map(
            all_terms,
            replace_subject(trials_r[0], 0))
        Lru = Lru.label_map(
            all_terms,
            replace_subject(trials_r[1], 1))
        Lru = Lru.label_map(
            all_terms,
            replace_test_function(tests_r[0]))

        Lrh = Lh_form.label_map(
            all_terms,
            replace_subject(trials_r[0], 0))
        Lrh = Lrh.label_map(
            all_terms,
            replace_subject(trials_r[1], 1))
        Lrh = Lrh.label_map(
            all_terms,
            replace_test_function(tests_r[1]))

        Liu = Lu_form.label_map(
            all_terms,
            replace_subject(trials_i[0], 0))
        Liu = Liu.label_map(
            all_terms,
            replace_subject(trials_i[1], 1))
        Liu = Liu.label_map(
            all_terms,
            replace_test_function(tests_i[0]))

        Lih = Lh_form.label_map(
            all_terms,
            replace_subject(trials_i[0], 0))
        Lih = Lih.label_map(
            all_terms,
            replace_subject(trials_i[1], 1))
        Lih = Lih.label_map(
            all_terms,
            replace_test_function(tests_i[1]))


        a = (
            self.ar * aur_mass_form
            + self.ar * aui_mass_form
            + self.ai * aur_mass_form
            - self.ai * aui_mass_form
            - self.tau * (Lru + Lrh)
            + self.ar * ahr_mass_form
            + self.ar * ahi_mass_form
            + self.ai * ahr_mass_form
            - self.ai * ahi_mass_form
            - self.tau * (Liu + Lih)
            + self.ai * aui_mass_form
        )

        L = (
            ur_mass_form.label_map(
                all_terms,
                replace_subject(ur, 0))
            + ui_mass_form.label_map(
                all_terms,
                replace_subject(ui, 0))
            + hr_mass_form.label_map(
                all_terms,
                replace_subject(hr, 1))
            + hi_mass_form.label_map(
                all_terms,
                replace_subject(hi, 1))
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
        u_in, D_in = U0.split()
        u0, _, h0, _ = self.U0.split()
        u0.assign(u_in)
        h0.assign(D_in)
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

        u_sum, _, h_sum, _ = self.w_sum.split()
        u_out, h_out = self.w_out.split()
        u_out.assign(u_sum)
        h_out.assign(h_sum)
        return self.w_out
