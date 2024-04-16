from firedrake import (Function, TrialFunctions, LinearVariationalProblem,
                       LinearVariationalSolver, DirichletBC, Constant)
from firedrake.fml import all_terms, drop, replace_subject, Term
from gusto.labels import (time_derivative, transport, prognostic,
                          transporting_velocity)
from gusto.rexi import Rexi
import numpy as np
import ufl


class AveragedModel(object):

    def __init__(self, domain, params):

        self.dt = Constant(60*60)
        self.params = params
        self.nlevels = 1

        Mbar = 9
        self.svals = np.arange(0.5, Mbar)/Mbar
        weights = np.exp(-1.0/self.svals/(1-self.svals))
        self.weights = weights / np.sum(weights)
        self.svals -= 0.5

    def setup(self, equation, apply_bcs=True, *active_labels):
        self.fs = equation.function_space
        self.x_Nin = Function(self.fs)
        self.x_Nout = Function(self.fs)
        self.x_out = Function(self.fs)
        self.exp = Rexi(equation, self.params)

        trials = TrialFunctions(self.fs)
        bcs = [DirichletBC(self.fs.sub(0), bc.function_arg,
                           bc.sub_domain) for bc in equation.bcs['u']]

        a = equation.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=replace_subject(trials),
            map_if_false=drop)

        self.residual = equation.residual.label_map(
            lambda t: t.has_label(transport) and t.get(prognostic) == "u",
            map_if_true=replace_subject(self.x_Nin),
            map_if_false=drop)

        self.residual += equation.residual.label_map(
            lambda t: t.has_label(transport) and t.get(prognostic) == "D",
            map_if_true=replace_subject(self.x_Nin),
            map_if_false=drop)

        self.residual -= equation.residual.label_map(
            lambda t: t.has_label(transport) and t.get(prognostic) == "D",
            map_if_true=replace_subject(equation.prescribed_fields('topography'), old_idx=1),
            map_if_false=drop)

        self.residual = self.residual.label_map(
            all_terms,
            lambda t: -self.dt*t)

        print("in avg, this is N:")
        for t in self.residual:
            print(t.form)

        nprob = LinearVariationalProblem(a.form, self.residual.form,
                                         self.x_Nout, bcs)
        self.nsolver = LinearVariationalSolver(nprob)

    def setup_transporting_velocity(self, uadv):
        self.residual = self.residual.label_map(
            lambda t: t.has_label(transporting_velocity),
            map_if_true=lambda t:
            Term(ufl.replace(t.form, {t.get(transporting_velocity): uadv}),
                 t.labels)
        )

        self.residual = transporting_velocity.update_value(self.residual, uadv)

    def apply(self, x_out, x_in):
        pass


class AveragedRK4(AveragedModel):

    def setup(self, equation, apply_bcs=True, *active_labels):
        super().setup(equation)
        self.nStages = 4
        self.V = [Function(self.fs) for i in range(self.nStages)]
        self.U = [Function(self.fs) for i in range(self.nStages)]

    def solve_stage(self, x0, stage):

        self.V[stage].assign(0.)

        for (s, weight) in zip(self.svals, self.weights):
            print(s, weight)
            self.exp.solve(self.x_Nin, x0, -s)
            if stage in [1, 3]:
                self.exp.solve(self.x_Nin, self.x_Nin, -self.dt/2)
            self.nsolver.solve()
            self.exp.solve(self.x_out, self.x_Nout, s)
            self.x_out *= weight
            self.V[stage] += self.x_out

        if stage == 0:
            self.U[stage].assign(x0 + 0.5*self.V[stage])
        elif stage == 1:
            self.exp.solve(self.x_out, x0, -self.dt/2)
            self.U[stage].assign(self.x_out - 0.5*self.V[stage])
        elif stage == 2:
            self.exp.solve(self.x_out, x0, -self.dt/2)
            self.U[stage].assign(self.x_out - 0.5*self.V[stage])
        elif stage == 3:
            self.exp.solve(self.x_out, x0, -self.dt)
            self.U[stage].assign(self.x_out)
            self.exp.solve(self.x_out, self.V[0], -self.dt)
            self.U[stage] += self.x_out / 6
            self.V[1] += self.V[2]
            self.exp.solve(self.x_out, self.V[1], -self.dt/2)
            self.U[stage] += self.x_out / 3
            self.U[stage] += self.V[3] / 6

    def apply(self, x_out, x_in):

        for stage in range(self.nStages):
            print("stage: ", stage)
            self.solve_stage(x_in, stage)
        x_out.assign(self.U[3])
