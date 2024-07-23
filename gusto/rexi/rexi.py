from gusto.rexi.rexi_coefficients import *
from firedrake import Function, TrialFunction, TestFunction, \
    Constant, DirichletBC, \
    LinearVariationalProblem, LinearVariationalSolver
from gusto.labels import time_derivative, prognostic, linearisation
from firedrake.fml import (
    Term, all_terms, drop, subject,
    replace_subject, replace_test_function, replace_trial_function
)
from firedrake.formmanipulation import split_form
from asQ.complex_proxy import vector as cpx


NullTerm = Term(None)


class Rexi(object):
    """
    Class defining the solver for the system

    (A_n + tau L)V_n = U

    required for computing the matrix exponential.
    """

    def __init__(self, equation, rexi_parameters, *, solver_parameters=None,
                 manager=None):

        """
        Args:
            equation (:class:`PrognosticEquation`): the model's equation
            rexi_parameters (:class:`RexiParameters`): Rexi configuration
                parameters
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the solver. Defaults to None.
            manager (:class:`.Ensemble`): the space and ensemble sub-
                communicators. Defaults to None.
        """
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

        # set dummy constants for tau and A_i and Beta
        self.br = Constant(1.)
        self.bi = Constant(1.)
        self.ar = Constant(1.)
        self.ai = Constant(1.)
        self.tau = Constant(1.)

        # set up functions, problem and solver
        W_ = equation.function_space
        W = cpx.FunctionSpace(W_)

        self.U0 = Function(W).assign(0)
        self.w = Function(W)
        self.wrk = Function(W_)

        test = TestFunction(W)
        trial = TrialFunction(W)
        tests_r = cpx.split(test, cpx.re)
        tests_i = cpx.split(test, cpx.im)
        trials_r = cpx.split(trial, cpx.re)
        trials_i = cpx.split(trial, cpx.im)

        U0r = cpx.subfunctions(self.U0, cpx.re)

        ar, ai = self.ar, self.ai
        a = NullTerm
        b = NullTerm

        for i in range(len(W_)):
            # residual only for prognostic i
            ith_res = residual.label_map(
                lambda t: t.get(prognostic) == equation.field_names[i],
                lambda t: Term(
                    split_form(t.form)[i].form,
                    t.labels),
                map_if_false=drop)

            # mass matrix terms for real/imaginary components
            mass_form = ith_res.label_map(
                lambda t: t.has_label(time_derivative),
                map_if_false=drop)

            mr = mass_form.label_map(
                all_terms,
                replace_test_function(tests_r[i]))

            mi = mass_form.label_map(
                all_terms,
                replace_test_function(tests_i[i]))

            # linear operator for real/imaginary components
            L_form = ith_res.label_map(
                lambda t: t.has_label(time_derivative),
                drop)

            Lr = L_form.label_map(
                all_terms,
                replace_test_function(tests_r[i]))

            Li = L_form.label_map(
                all_terms,
                replace_test_function(tests_i[i]))

            # real matrix M multiplied by complex number (ar + i*ai)
            a += (
                + ar * mr.label_map(all_terms, replace_subject(trials_r[i], old_idx=i))
                - ai * mr.label_map(all_terms, replace_subject(trials_i[i], old_idx=i))
            )

            a += (
                + ai * mi.label_map(all_terms, replace_subject(trials_r[i], old_idx=i))
                + ar * mi.label_map(all_terms, replace_subject(trials_i[i], old_idx=i))
            )

            # real matrix M multiplied by real number tau
            a -= self.tau * Lr.label_map(all_terms, replace_subject(trials_r))
            a -= self.tau * Li.label_map(all_terms, replace_subject(trials_i))

            # right hand side only has real valued component
            b += mr.label_map(all_terms, replace_subject(U0r[i], i))

        a = a.label_map(lambda t: t is NullTerm, drop)
        b = b.label_map(lambda t: t is NullTerm, drop)

        if hasattr(equation, "aP"):
            aP = equation.aP(trial, self.ai, self.tau)
        else:
            aP = None

        # BCs are declared for the plain velocity space.
        # First we need to transfer the velocity boundary conditions to the
        # velocity component of the mixed space.
        uidx = equation.field_names.index('u')
        ubcs = (DirichletBC(W_.sub(uidx), bc.function_arg, bc.sub_domain)
                for bc in equation.bcs['u'])

        # now we can transfer the velocity boundary conditions to the complex space
        bcs = tuple(cb for bc in ubcs for cb in cpx.DirichletBC(W, W_, bc))

        rexi_prob = LinearVariationalProblem(a.form, b.form, self.w, aP=aP,
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
        cpx.set_real(self.U0, x_in)
        x_out.assign(0.)

        # loop over solvers, assigning a_i, solving and accumulating the sum
        for i in range(self.N):
            j = self.idx + i
            self.ar.assign(self.alpha[j].real)
            self.ai.assign(self.alpha[j].imag)

            self.br.assign(self.beta[j].real)
            self.bi.assign(self.beta[j].imag)

            self.solver.solve()

            cpx.get_real(self.w, self.wrk)
            x_out += self.br*self.wrk

            cpx.get_imag(self.w, self.wrk)
            x_out -= self.bi*self.wrk

        # in parallel we have to accumulate the sum over all processes
        if self.manager is not None:
            self.wrk.assign(x_out)
            self.manager.allreduce(self.wrk, x_out)
