from gusto import(
    ShallowWaterParameters, Domain, ShallowWaterEquations,rtheta_from_xy,
    CoriolisOptions
)
from firedrake import (
    SpatialCoordinate, PeriodicRectangleMesh, conditional, FunctionSpace,
    Function
)
import sympy as sp

def smooth_f_profile(delta, rstar, Omega, R, Lx, nx):
    delta *= Lx/nx
    r = sp.symbols('r')
    fexpr = 2*Omega*(1-0.5*r**2/R**2)
    left_val = fexpr.subs(r, rstar-delta)
    right_val = 2*Omega
    left_diff_val = sp.diff(fexpr, r).subs(r, rstar-delta)
    left_diff2_val = sp.diff(fexpr, r, 2).subs(r, rstar-delta)

    a = sp.symbols(f'a_0:{6}')
    P = a[0]
    for i in range(1, 6):
        P += a[i]*r**i

    eqns = [
        P.subs(r, rstar-delta) - left_val,
        P.subs(r, rstar+delta) - right_val,
        sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
        sp.diff(P, r).subs(r, rstar+delta),
        sp.diff(P, r, 2).subs(r, rstar-delta) - left_diff2_val,
        sp.diff(P, r, 2).subs(r, rstar+delta)
    ]

    sol = sp.solve(eqns, a)
    coeffs = [sol[sp.Symbol(f'a_{i}')] for i in range(6)]
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
fexpr = 2*Omega*(1-0.5*r*2/R**2)

coeffs = smooth_f_profile(delta=smooth_delta, rstar=rstar, Omega=Omega_num, R=R, Lx=Lx, nx=nx)
fsmooth = float(coeffs[0]) + float(coeffs[1])*r + float(coeffs[2])*r**2 + float(coeffs[3])*r**3
fsmooth += float(coeffs[4])*r**4 + float(coeffs[5])*r**5

Vcg = FunctionSpace(domain.mesh, "DG", 1)

trap = 'no_trap'

if trap=='no_trap':
    pv_true = Function(Vcg).interpolate(fexpr)

    eqns = ShallowWaterEquations(domain, parameters)
    gusto_field = eqns.prescribed_fields('coriolis')
    pv_gusto = Function(Vcg).interpolate(gusto_field)

elif trap=='step':
    ftrap_step = conditional(r<rstar, fexpr, 2*Omega)
    pv_true = Function(Vcg).interpolate(ftrap_step)

    eqns_step = ShallowWaterEquations(domain, parameters, coriolis_trap=(rstar, 2*Omega))
    gusto_step_field = eqns_step.prescribed_fields('coriolis')
    pv_gusto = Function(Vcg).interpolate(gusto_step_field)

elif trap=='smooth':
    ftrap1 = conditional(r<rstar-smooth_delta*Lx/nx, fexpr, fsmooth)
    ftrap_smooth = conditional(r<rstar+smooth_delta*Lx/nx, ftrap1, 2*Omega)
    pv_true = Function(Vcg).interpolate(ftrap_smooth)

    eqns_smooth = ShallowWaterEquations(domain, parameters, coriolis_trap=(rstar-smooth_delta*Lx/nx, ftrap_smooth))
    gusto_smooth_field = eqns_smooth.prescribed_fields('coriolis')
    pv_gusto = Function(Vcg).interpolate(gusto_smooth_field)