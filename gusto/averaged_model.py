from firedrake import Function
from gusto.fml.form_manipulation_labelling import Term, keep, all_terms
from gusto.labels import linearisation, nonlinear, time_derivative, replace_trial_function, subject
import numpy as np

class AveragedModel(object):

    nlevels = 1

    def __init__(self, eta, Mbar, exponential_method, exponential_method2,
                 timestepping_scheme):

        self.eta = eta
        svals = np.arange(0.5, Mbar)/Mbar
        weights = np.exp(-1.0/svals/(1.0-svals))
        self.weights = weights/np.sum(weights)
        self.svals = svals - 0.5

        self.exp = exponential_method
        self.exp2 = exponential_method2
        self.stepper = timestepping_scheme
        self.dt = timestepping_scheme.dt

    def setup(self, equation, ubar):

        def nonlinearise(t):
            t_linear = Term(t.get(linearisation).form, t.labels)
            t_linear = replace_trial_function(t.get(subject))(t_linear)
            return nonlinear(t, Term(t.form-t_linear.form, t.labels))

        # set up nonlinear residual
        equation.residual = equation.residual.label_map(
            lambda t: t.has_label(linearisation) and not t.has_label(time_derivative),
            map_if_true=lambda t: nonlinearise(t),
            map_if_false=lambda t: nonlinear(t, Term(t.form, t.labels)))

        # set up functions to store input and output to forward map
        W = equation.function_space
        self.exp_in = Function(W)
        self.exp_out = Function(W)

        self.V = Function(W)
        self.V_out = Function(W)
        self.x_out = Function(W)

        self.stepper.setup(equation, ubar, True, nonlinear)

    def apply(self, x_out, x_in):

        self.exp_in.assign(x_in)

        # loop over weights
        for k in range(len(self.svals)):

            expt = self.eta*self.dt*self.svals[k]

            # apply forward map
            self.exp.apply(self.exp_out, self.exp_in, expt)
            self.V.assign(self.exp_out)

            # timestep V
            self.stepper.apply(self.V_out, self.V)

            # compute difference from Un
            self.V.assign(self.V_out)
            self.V -= self.exp_out

            # apply backwards map
            self.exp.apply(self.exp_out, self.V, -expt)

            # multiply by weight and add to total
            self.exp_out *= self.weights[k]

            x_in += self.exp_out

        self.x_out.assign(x_in)

        self.exp2.apply(x_out, self.x_out, self.dt)
