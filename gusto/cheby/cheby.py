from firedrake import (Function, TrialFunctions, Constant, real, imag,
                       LinearVariationalProblem, LinearVariationalSolver)
from gusto.fml.form_manipulation_labelling import drop, all_terms, Term
from gusto.labels import (linearisation, time_derivative,
                          replace_subject, replace_trial_function)
import numpy as np
from scipy import fftpack

from firedrake.petsc import PETSc
print = PETSc.Sys.Print

__all__ = ["Cheby"]


class Cheby(object):

    def __init__(self, equation, ncheb, tol, L, filter_val=None):
        """
        Class to apply the exponential of an operator
        using chebyshev approximation

        operator_solver: a VariationalLinearSolver implementing
        the forward operator for Shift-and-Invert (used for residual
        calculation)
        operator_in: the input to operator_solver
        operator_out: the output to operator_solver
        ncheb: number of Chebyshev polynomials to approximate exp
        tol: tolerance to compress Chebyshev expansion by
        (removes terms from the high degree end until total L^1 norm
        of removed terms > tol)
        L: approximate exp on range [-L*i, L*i]
        """

        # construct solver from equation residual
        residual = equation.residual.label_map(
            lambda t: t.has_label(linearisation),
            map_if_true=lambda t: Term(t.get(linearisation).form, t.labels),
            map_if_false=drop)

        W = equation.function_space
        trials = TrialFunctions(W)
        self.x_in = Function(W)
        self.x_out = Function(W)

        a = residual.label_map(lambda t: t.has_label(time_derivative),
                               replace_subject(trials),
                               drop)
        F = residual.label_map(lambda t: t.has_label(time_derivative),
                               map_if_true=drop,
                               map_if_false=lambda t: -1*t)
        F = F.label_map(lambda t: all_terms, replace_trial_function(self.x_in))

        params = {
            'ksp_type': 'preonly',
            'pc_type': 'fieldsplit',
            'fieldsplit_0_ksp_type':'cg',
            'fieldsplit_0_pc_type':'bjacobi',
            'fieldsplit_0_sub_pc_type':'ilu',
            'fieldsplit_1_ksp_type':'preonly',
            'fieldsplit_1_pc_type':'bjacobi',
            'fieldsplit_1_sub_pc_type':'ilu'
        }

        cheby_prob = LinearVariationalProblem(a.form, F.form, self.x_out)
        self.solver = LinearVariationalSolver(cheby_prob,
                                              solver_parameters=params)

        dpi = np.pi/(ncheb+1)
        t1 = np.arange(np.pi, -dpi/2, -dpi)
        x = L*np.cos(t1)
        fvals = np.exp(1j*x)

        # Set cut-off frequency
        eigs = [0.003465, 0.007274, 0.014955] # maximum frequency
        fL = eigs[0]*60*60

        if filter_val is not None:
            fvals /= (1 + (x/(filter_val*L))**2)**4

        valsUnitDisc = np.concatenate((np.flipud(fvals), fvals[1:-1]))
        FourierCoeffs = fftpack.fft(valsUnitDisc)/ncheb

        self.ChebCoeffs = FourierCoeffs[:ncheb+2]
        self.ChebCoeffs[0] = self.ChebCoeffs[0]/2
        self.ChebCoeffs[-1] = self.ChebCoeffs[-1]/2

        #cheby compression
        nrm = 0.
        Compressed = False
        while nrm + abs(self.ChebCoeffs[ncheb+1]) < tol:
            nrm += abs(self.ChebCoeffs[ncheb+1])
            ncheb -= 1
            Compressed = True
        assert Compressed
        self.ncheb = ncheb

        #initialise T0
        A = 0
        Tnm1 = 1.0
        Tn = A/(L*1j)
        fvals0 = self.ChebCoeffs[0]*Tnm1 + self.ChebCoeffs[1]*Tn

        for i in range(2,ncheb+1):
            Tnm2 = Tnm1
            Tnm1 = Tn
            Tn = 2*A*Tnm1/(L*1j) - Tnm2
            fvals0 += self.ChebCoeffs[i]*Tn

        for i in range(len(self.ChebCoeffs)):
            self.ChebCoeffs[i] = self.ChebCoeffs[i]/fvals0

        #check if fvals0 = 1
        Tnm1 = 1.0
        Tn = A/(L*1j)
        fvals0 = self.ChebCoeffs[0]*Tnm1 + self.ChebCoeffs[1]*Tn

        for i in range(2,ncheb+1):
            Tnm2 = Tnm1
            Tnm1 = Tn
            Tn = 2*A*Tnm1/(L*1j) - Tnm2
            fvals0 += self.ChebCoeffs[i]*Tn

        self.L = L

        self.Tm1_r = Function(W)
        self.Tm1_i = Function(W)
        self.Tm2_r = Function(W)
        self.Tm2_i = Function(W)
        self.T_r = Function(W)
        self.T_i = Function(W)

        self.dy = Function(W)

    def apply(self, x_out, x_in, t):
        L = self.L
        #initially Tm1 contains T_0(A)x
        #T_0(x) = x^0 i.e. T_0(tA) = I, T_0(tA)x = x
        self.Tm1_r.assign(x_in)
        self.Tm1_i.assign(0)
        
        Coeff = Constant(1)

        x_out.assign(0.)
        self.dy.assign(self.Tm1_r)
        Coeff.assign(np.real(self.ChebCoeffs[0]))
        self.dy *= Coeff
        x_out += self.dy
        
        #initially T contains T_1(tA)x
        #T_1(x) = x^1/(i*L) i.e. T_1(tA) = -i*tA/L, T_1(tA)x = -i*tAx/L
        self.x_in.assign(x_in)
        self.solver.solve()
        self.T_r.assign(0)
        self.T_i.assign(self.x_out)
        self.T_i *= -t/L

        self.dy.assign(self.T_i)
        Coeff.assign(np.imag(self.ChebCoeffs[1]))
        self.dy.assign(-Coeff*self.dy)
        x_out += self.dy

        for i in range(2, self.ncheb+1):
            self.Tm2_r.assign(self.Tm1_r)
            self.Tm2_i.assign(self.Tm1_i)
            self.Tm1_r.assign(self.T_r)
            self.Tm1_i.assign(self.T_i)

            #Tn = 2*t*A*Tnm1/(L*1j) - Tnm2
            self.x_in.assign(self.Tm1_r)
            self.solver.solve()
            self.T_i.assign(self.x_out)
            self.T_i *= -2*t/L
            self.x_in.assign(self.Tm1_i)
            self.solver.solve()
            self.T_r.assign(self.x_out)
            self.T_r *= 2*t/L

            self.T_i -= self.Tm2_i
            self.T_r -= self.Tm2_r

            self.dy.assign(self.T_r)
            Coeff.assign(real(self.ChebCoeffs[i]))
            self.dy *= Coeff
            x_out += self.dy

            self.dy.assign(self.T_i)
            Coeff.assign(imag(self.ChebCoeffs[i]))
            self.dy.assign(-Coeff*self.dy)
            x_out += self.dy
