from gusto.rexi.rexi_coefficients import *
from firedrake import Function, TrialFunctions, TestFunctions, \
    Constant, DirichletBC, \
    LinearVariationalProblem, LinearVariationalSolver, MixedFunctionSpace
from gusto.labels import time_derivative, prognostic, linearisation
from firedrake.fml import (
    Term, all_terms, drop, subject,
    replace_subject, replace_test_function, replace_trial_function
)
from firedrake.formmanipulation import split_form


NullTerm = Term(None)


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

        residual = equation.residual.label_map(
            lambda t: t.has_label(linearisation),
            map_if_true=lambda t: Term(t.get(linearisation).form, t.labels),
            map_if_false=drop)
        residual = residual.label_map(
            all_terms,
            lambda t: replace_trial_function(t.get(subject))(t))

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
        spaces = []
        for i in range(len(W_)):
            spaces.append(W_[i])
            spaces.append(W_[i])
        W = MixedFunctionSpace(spaces)
        self.U0 = Function(W)
        self.w_sum = Function(W)
        self.w = Function(W)
        self.w_ = Function(W)
        tests = TestFunctions(W)
        trials = TrialFunctions(W)
        tests_r = tests[::2]
        tests_i = tests[1::2]
        trials_r = trials[::2]
        trials_i = trials[1::2]

        ar, ai = self.ar, self.ai
        a = NullTerm
        L = NullTerm
        for i in range(len(W_)):
            ith_res = residual.label_map(
                lambda t: t.get(prognostic) == equation.field_names[i],
                lambda t: Term(
                    split_form(t.form)[i].form,
                    t.labels),
                map_if_false=drop)

            mass_form = ith_res.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_false=drop)

            m = mass_form.label_map(
                all_terms,
                replace_test_function(tests_r[i]))
            a += (
                (ar + ai) * m.label_map(all_terms,
                                        replace_subject(trials_r[i], old_idx=i))
                + (ar - ai) * m.label_map(all_terms,
                                          replace_subject(trials_i[i], old_idx=i))
            )

            L += (
                m.label_map(all_terms, replace_subject(self.U0.subfunctions[2*i], i))
                + m.label_map(all_terms, replace_subject(self.U0.subfunctions[2*i+1], old_idx=i))
            )

            m = mass_form.label_map(
                all_terms,
                replace_test_function(tests_i[i]))
            a += (
                (ar - ai) * m.label_map(all_terms,
                                        replace_subject(trials_r[i], old_idx=i))
                + (-ar - ai) * m.label_map(all_terms,
                                           replace_subject(trials_i[i], old_idx=i))
            )

            L += (
                m.label_map(all_terms,
                            replace_subject(self.U0.subfunctions[2*i], i))
                - m.label_map(all_terms,
                              replace_subject(self.U0.subfunctions[2*i+1], i))
            )

            L_form = ith_res.label_map(
                lambda t: t.has_label(time_derivative),
                drop)

            Lr = L_form.label_map(
                all_terms,
                replace_test_function(tests_r[i]))
            a -= self.tau * Lr.label_map(all_terms,
                                         replace_subject(trials_r))
            a -= self.tau * Lr.label_map(all_terms,
                                         replace_subject(trials_i))

            Li = L_form.label_map(
                all_terms,
                replace_test_function(tests_i[i]))
            a -= self.tau * Li.label_map(all_terms,
                                         replace_subject(trials_r))
            a += self.tau * Li.label_map(all_terms,
                                         replace_subject(trials_i))

        a = a.label_map(lambda t: t is NullTerm, drop)
        L = L.label_map(lambda t: t is NullTerm, drop)

        if hasattr(equation, "aP"):
            aP = equation.aP(trial, self.ai, self.tau)
        else:
            aP = None

        # Boundary conditions (assumes extruded mesh)
        # BCs are declared for the plain velocity space. As we need them in
        # extended mixed problem, we replicate the BCs but for subspace of W
        bcs = []
        for bc in equation.bcs['u']:
            bcs.append(DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain))
            bcs.append(DirichletBC(W.sub(1), bc.function_arg, bc.sub_domain))

        rexi_prob = LinearVariationalProblem(a.form, L.form, self.w, aP=aP,
                                             bcs=bcs,
                                             constant_jacobian=False)

        # if solver_parameters is None:
        #    solver_parameters = equation.solver_parameters

        self.solver = LinearVariationalSolver(
            rexi_prob, solver_parameters=solver_parameters)

    def solve(self, x_out, x_in, dt):
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
        Uin = x_in.subfunctions
        U0 = self.U0.subfunctions
        for i in range(len(Uin)):
            U0[2*i].assign(Uin[i])
        self.w_.assign(0.)
        w_ = self.w_.subfunctions
        w = self.w.subfunctions

        # loop over solvers, assigning a_i, solving and accumulating the sum
        for i in range(self.N):
            j = self.idx + i
            self.ar.assign(self.alpha[j].real)
            self.ai.assign(self.alpha[j].imag)
            self.solver.solve()
            for k in range(len(Uin)):
                wk = w_[2*k]
                wk += Constant(self.beta[j].real)*w[2*k] - Constant(self.beta[j].imag)*w[2*k+1]

        # in parallel we have to accumulate the sum over all processes
        if self.manager is not None:
            self.manager.allreduce(self.w_, self.w_sum)
        else:
            self.w_sum.assign(self.w_)

        w_sum = self.w_sum.subfunctions
        w_out = self.w_out.subfunctions
        for i in range(len(w_out)):
            w_out[i].assign(w_sum[2*i])

        x_out.assign(self.w_out)
