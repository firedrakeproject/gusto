from firedrake import *
import numpy as np
from scipy import fftpack

from firedrake.petsc import PETSc
print = PETSc.Sys.Print

__all__ = ["cheby_exp"]

class cheby_exp(object):
    def __init__(self, operator_solver, operator_in, operator_out,
                 ncheb, tol, L, filter=False, filter_val=0.75, filter_freq=False):
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

        self.operator_solver = operator_solver
        self.operator_in = operator_in
        self.operator_out = operator_out

        dpi = np.pi/(ncheb+1)
        t1 = np.arange(np.pi, -dpi/2, -dpi)
        x = L*np.cos(t1)
        fvals = np.exp(1j*x)

        #Set cut-off frequency
        eigs = [0.003465, 0.007274, 0.014955] #maximum frequency
        fL = eigs[0]*60*60

        print("L =", L)
        if filter_freq:
            print("filter_freq is on. L =", fL)
            fvals /= (1 + (x/fL)**2)**4
        elif filter:
            print("filter is on. L =", filter_val*L)
            fvals /= (1 + (x/(filter_val*L))**2)**4

        valsUnitDisc = np.concatenate((np.flipud(fvals), fvals[1:-1]))
        FourierCoeffs = fftpack.fft(valsUnitDisc)/ncheb

        self.ChebCoeffs = FourierCoeffs[:ncheb+2]
        self.ChebCoeffs[0] = self.ChebCoeffs[0]/2
        self.ChebCoeffs[-1] = self.ChebCoeffs[-1]/2

        #cheby compression
        print("ncheb before compression", ncheb)
        nrm = 0.
        Compressed = False
        while nrm + abs(self.ChebCoeffs[ncheb+1]) < tol:
            nrm += abs(self.ChebCoeffs[ncheb+1])
            ncheb -= 1
            Compressed = True
        assert Compressed
        print("ncheb after compression", ncheb)
        self.ncheb = ncheb
        print("ncheb is set to", ncheb)

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

        print("fvals0 before initialisation", fvals0)

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

        print("fvals0 after initialisation", fvals0)

        self.L = L

        FS = operator_in.function_space()
        self.Tm1_r = Function(FS)
        self.Tm1_i = Function(FS)
        self.Tm2_r = Function(FS)
        self.Tm2_i = Function(FS)
        self.T_r = Function(FS)
        self.T_i = Function(FS)

        self.dy = Function(FS)

    def apply(self, x, y, t):
        L = self.L
        #initially Tm1 contains T_0(A)x
        #T_0(x) = x^0 i.e. T_0(tA) = I, T_0(tA)x = x
        self.Tm1_r.assign(x)
        self.Tm1_i.assign(0)
        
        Coeff = Constant(1)

        y.assign(0.)
        self.dy.assign(self.Tm1_r)
        Coeff.assign(np.real(self.ChebCoeffs[0]))
        self.dy *= Coeff
        y += self.dy
        
        #initially T contains T_1(tA)x
        #T_1(x) = x^1/(i*L) i.e. T_1(tA) = -i*tA/L, T_1(tA)x = -i*tAx/L
        self.operator_in.assign(x)
        self.operator_solver.solve()
        self.T_r.assign(0)
        self.T_i.assign(self.operator_out)
        self.T_i *= -t/L

        self.dy.assign(self.T_i)
        Coeff.assign(np.imag(self.ChebCoeffs[1]))
        self.dy.assign(-Coeff*self.dy)
        y += self.dy

        for i in range(2, self.ncheb+1):
            self.Tm2_r.assign(self.Tm1_r)
            self.Tm2_i.assign(self.Tm1_i)
            self.Tm1_r.assign(self.T_r)
            self.Tm1_i.assign(self.T_i)

            #Tn = 2*t*A*Tnm1/(L*1j) - Tnm2
            self.operator_in.assign(self.Tm1_r)
            self.operator_solver.solve()
            self.T_i.assign(self.operator_out)
            self.T_i *= -2*t/L
            self.operator_in.assign(self.Tm1_i)
            self.operator_solver.solve()
            self.T_r.assign(self.operator_out)
            self.T_r *= 2*t/L

            self.T_i -= self.Tm2_i
            self.T_r -= self.Tm2_r

            self.dy.assign(self.T_r)
            Coeff.assign(real(self.ChebCoeffs[i]))
            self.dy *= Coeff
            y += self.dy

            self.dy.assign(self.T_i)
            Coeff.assign(imag(self.ChebCoeffs[i]))
            self.dy.assign(-Coeff*self.dy)
            y += self.dy
