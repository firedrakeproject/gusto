from gusto.rexi.rexi_coefficients import *
from firedrake import Function, DirichletBC, \
    LinearVariationalProblem, LinearVariationalSolver
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

    required for computing the matrix exponential.
    """

    def __init__(self, equation, rexi_parameters, *, solver_parameters=None,
                 manager=None, cpx_type='mixed'):

        """
        Args:
            equation (:class:`PrognosticEquation`): the model's equation
            rexi_parameters (:class:`RexiParameters`): Rexi configuration
                parameters
            solver_parameters (dict, optional): dictionary of parameters to
                pass to the solver. Defaults to None.
            manager (:class:`.Ensemble`): the space and ensemble sub-
                communicators. Defaults to None.
            cpx_type (str, optional): implementation of complex-valued space,
                can be 'mixed' or 'vector'.
        """
        if cpx_type == 'mixed':
            from gusto.complex_proxy import mixed as cpx
        elif cpx_type == 'vector':
            from gusto.complex_proxy import vector as cpx
        else:
            raise ValueError("cpx_type must be 'mixed' or 'vector'")
        self.cpx = cpx

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

        # set up complex function space
        W_ = equation.function_space
        W = cpx.FunctionSpace(W_)

        self.U0 = Function(W_)   # right hand side function
        self.w = Function(W)     # solution
        self.wrk = Function(W_)  # working buffer

        ncpts = len(W_)

        # split equation into mass matrix and linear operator
        mass = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_false=drop)

        function = residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=drop)

        # generate ufl for mass matrix over given trial/tests
        def form_mass(*trials_and_tests):
            trials = trials_and_tests[:ncpts]
            tests = trials_and_tests[ncpts:]
            m = mass.label_map(
                all_terms,
                replace_test_function(tests))
            m = m.label_map(
                all_terms,
                replace_subject(trials))
            return m

        # generate ufl for linear operator over given trial/tests
        def form_function(*trials_and_tests):
            trials = trials_and_tests[:ncpts]
            tests = trials_and_tests[ncpts:]

            f = NullTerm
            for i in range(ncpts):
                fi = function.label_map(
                    lambda t: t.get(prognostic) == equation.field_names[i],
                    lambda t: Term(
                        split_form(t.form)[i].form,
                        t.labels),
                    map_if_false=drop)

                fi = fi.label_map(
                    all_terms,
                    replace_test_function(tests[i]))

                fi = fi.label_map(
                    all_terms,
                    replace_subject(trials))

                f += fi
            f = f.label_map(lambda t: t is NullTerm, drop)
            return f

        # generate ufl for right hand side over given trial/tests
        def form_rhs(*tests):
            rhs = mass.label_map(
                all_terms,
                replace_test_function(tests))
            rhs = rhs.label_map(
                all_terms,
                replace_subject(self.U0))
            return rhs

        # complex Constants for alpha and beta values
        self.ac = cpx.ComplexConstant(1)
        self.bc = cpx.ComplexConstant(1)

        # alpha*M and tau*L
        aM = cpx.BilinearForm(W, self.ac, form_mass)
        aL, self.tau, _ = cpx.BilinearForm(W, 1, form_function, return_z=True)
        a = aM - aL

        # right hand side is just U0
        b = cpx.LinearForm(W, 1, form_rhs)

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

        :arg x_in: the mixed function on the rhs.
        :arg dt: the value of tau

        """
        cpx = self.cpx
        # assign tau and U0 and initialise solution to 0.
        self.tau.assign(dt)
        self.U0.assign(x_in)
        x_out.assign(0.)

        # loop over solvers, assigning a_i, solving and accumulating the sum
        for i in range(self.N):
            j = self.idx + i
            self.ac.real.assign(self.alpha[j].real)
            self.ac.imag.assign(self.alpha[j].imag)

            self.bc.real.assign(self.beta[j].real)
            self.bc.imag.assign(self.beta[j].imag)

            self.solver.solve()

            # accumulate real part of beta*w
            cpx.get_real(self.w, self.wrk)
            x_out += self.bc.real*self.wrk

            cpx.get_imag(self.w, self.wrk)
            x_out -= self.bc.imag*self.wrk

        # in parallel we have to accumulate the sum over all processes
        if self.manager is not None:
            self.wrk.assign(x_out)
            self.manager.allreduce(self.wrk, x_out)
