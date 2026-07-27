from gusto import(
    ShallowWaterParameters, Domain, ShallowWaterEquations,rtheta_from_xy,
    CoriolisOptions
)
from firedrake import (
    SpatialCoordinate, PeriodicRectangleMesh, conditional
)
import scipy
import numpy as np
import time
import os
import shutil
import sympy as sp

def smooth_f_profile(degree, delta, style, rstar, Omega, R, Lx, nx):
    delta *= Lx/nx
    r = sp.symbols('r')
    if style == 'polar':
        fexpr = 2*Omega*(1-0.5*r**2/R**2)
        left_val = fexpr.subs(r, rstar-delta)
        right_val = 2*Omega
        left_diff_val = sp.diff(fexpr, r).subs(r, rstar-delta)
        left_diff2_val = sp.diff(fexpr, r, 2).subs(r, rstar-delta)
    elif style == 'flat':
        left_val = 2*Omega*(1-0.5*(rstar-delta)**2/R**2)
        right_val = 2*Omega
        left_diff_val = 0
        left_diff2_val = 0

    a = sp.symbols(f'a_0:{degree+1}')
    P = a[0]
    for i in range(1, degree+1):
        P += a[i]*r**i

    if degree == 3:
        eqns = [
            P.subs(r, rstar-delta) - left_val,
            P.subs(r, rstar+delta) - right_val,
            sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
            sp.diff(P, r).subs(r, rstar+delta)
        ]
    elif degree == 5:
        eqns = [
            P.subs(r, rstar-delta) - left_val,
            P.subs(r, rstar+delta) - right_val,
            sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
            sp.diff(P, r).subs(r, rstar+delta),
            sp.diff(P, r, 2).subs(r, rstar-delta) - left_diff2_val,
            sp.diff(P, r, 2).subs(r, rstar+delta)
        ]
    else:
        print('do not have BCs for this degree')

    sol = sp.solve(eqns, a)
    coeffs = [sol[sp.Symbol(f'a_{i}')] for i in range(degree+1)]
    # P_smooth = P.subs(sol)
    # f_smooth = sp.Piecewise(
    #     (fexpr, r<rstar-delta),
    #     (P_smooth, (rstar-delta<=r) & (r<=rstar+delta)),
    #     (right_val, rstar+delta<r)
    # )
    return coeffs

nx = 256
ny = nx
Lx = 7e7
Ly = Lx

rstar = Lx/2-3*Lx/nx
smooth_delta = 2

Bu = 1
g = 24.79
Omega = 1.74e-4
R = 71.4e6
f0 = 2 * Omega
rm = 1e6
phi0 = Bu * (f0*rm)**2
H = phi0/g

smooth_degree = 5
smooth_delta = 2

dt = 250

mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly, quadrilateral=True)

parameters = ShallowWaterParameters(mesh, H=H, Omega=Omega, R=R,
                                    rotation=CoriolisOptions.gammaplane)

domain = Domain(mesh, dt, "RTCF", 1)


x, y = SpatialCoordinate(mesh)
r, _ = rtheta_from_xy(x, y, Lx/2, Ly/2)
Rsq = parameters.R**2

Omega_num = Omega
Omega = parameters.Omega

# expression for gamma plane with no trap
fexpr = 2*Omega*(1-0.5*r*2/R**2)

# calculate polymonial that smooths trap edge
coeffs = smooth_f_profile(degree=smooth_degree, delta=smooth_delta, style='polar', rstar=rstar, Omega=Omega_num, R=R, Lx=Lx, nx=nx)
fsmooth = float(coeffs[0]) + float(coeffs[1])*r + float(coeffs[2])*r**2 + float(coeffs[3])*r**3
if smooth_degree == 5:
    fsmooth += float(coeffs[4])*r**4 + float(coeffs[5])*r**5


### analytic gamma plane with step-edge-trap - gamma plane inside rstar, 2*Omega outside
ftrap_step = conditional(r<rstar, fexpr, 2*Omega)

### ShallowWaterEquations with step-edge-trap
eqns_step = ShallowWaterEquations(domain, parameters, coriolis_trap=(rstar, 2*Omega))

### analytic gamma plane with smooth-edge-trap - gamma plane inside rstar-2*dx, then smoothing polynomial, then 2*Omega
ftrap1 = conditional(r<rstar-smooth_delta*Lx/nx, fexpr, fsmooth)
ftrap_smooth = conditional(r<rstar+smooth_delta*Lx/nx, ftrap1, 2*Omega)

### ShallowWaterEquations with smooth-edge-trap
eqns_smooth = ShallowWaterEquations(domain, parameters, coriolis_trap=(rstar-smooth_delta*Lx/nx, ftrap_smooth))

breakpoint()