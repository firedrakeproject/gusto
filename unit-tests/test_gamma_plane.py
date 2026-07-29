from gusto import (
    ShallowWaterParameters, ShallowWaterEquations, rtheta_from_xy,
    CoriolisOptions, OutputParameters, SIQNModel, RelativeVorticity,
    PotentialVorticity
)
from firedrake import (
    SpatialCoordinate, PeriodicRectangleMesh, conditional, FunctionSpace,
    Function, tricontourf, errornorm
)
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import pytest


def smooth_f_profile(delta, rstar, Omega, R, Lx, nx):
    """
    This function does...??
    """
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
    return [float(coeff) for coeff in coeffs]


@pytest.mark.parametrize("trap", ["no_trap", "step", "smooth"])
def test_gamma_plane(tmpdir, trap):

    # Define mesh parameters and create mesh
    nx = 256
    ny = nx
    Lx = 7e7
    Ly = Lx
    mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly, quadrilateral=True)

    # Define physical parameters
    g = 24.79                # gravity
    Omega = 1.74e-4          # planetary rotation
    f0 = 2 * Omega           # Coriolis parameter
    R = 71.4e6               # planetary radius
    Bu = 1                   # Burger number
    rm = 1e6                 # ??
    phi0 = Bu * (f0*rm)**2   # ??
    H = phi0/g               # mean depth
    rstar = Lx/2-3*Lx/nx     # ??
    parameters = ShallowWaterParameters(mesh, H=H, Omega=Omega, R=R,
                                        rotation=CoriolisOptions.gammaplane)

    # Define timestep
    dt = 250

    # Define correct Coriolis expression
    x, y = SpatialCoordinate(mesh)
    r, _ = rtheta_from_xy(x, y, Lx/2, Ly/2)
    fexpr = 2*Omega*(1-0.5*r**2/R**2)

    # Set up CG function for correct Coriolis field, set analytically
    Vcg = FunctionSpace(mesh, "CG", 1)
    pv_true = Function(Vcg)

    if trap == 'no_trap':
        pv_true.interpolate(fexpr)
        coriolis_trap = None

    elif trap == 'step':
        ftrap_step = conditional(r < rstar, fexpr, 2*Omega)
        pv_true.interpolate(ftrap_step)
        coriolis_trap = (rstar, 2*Omega)

    elif trap == 'smooth':
        smooth_delta = 2
        coeffs = smooth_f_profile(delta=smooth_delta,
                                  rstar=rstar, Omega=Omega, R=R, Lx=Lx, nx=nx)
        fsmooth = (
            coeffs[0] + coeffs[1]*r + coeffs[2]*r**2
            + coeffs[3]*r**3 + coeffs[4]*r**4 + coeffs[5]*r**5
        )

        ftrap1 = conditional(r < rstar-smooth_delta*Lx/nx, fexpr, fsmooth)
        ftrap_smooth = conditional(
            r < rstar+smooth_delta*Lx/nx, ftrap1, 2*Omega
        )
        pv_true.interpolate(ftrap_smooth)
        coriolis_trap = (rstar-smooth_delta*Lx/nx, ftrap_smooth)

    # I/O
    output = OutputParameters(dirname=str(tmpdir), dump_vtus=True)

    diagnostic_fields = [RelativeVorticity(), PotentialVorticity()]

    eqns = ShallowWaterEquations

    model = SIQNModel(mesh, dt, parameters, eqns, family='RTCF',
                      coriolis_trap=coriolis_trap)
    model.setup(output, diagnostic_fields=diagnostic_fields)

    stepper = model.stepper
    D0 = stepper.fields('D')
    D0.assign(H)
    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    model.run(t=0, tmax=10*dt)

    pv_true /= H
    pv_gusto = model.stepper.fields('PotentialVorticity')

    fig, axes = plt.subplots(1, 2)
    levels1 = np.linspace(pv_true.dat.data.min(),
                          pv_true.dat.data.max(), 10)
    c1 = tricontourf(pv_true, levels=levels1, axes=axes[0])
    fig.colorbar(c1)
    levels2 = np.linspace(pv_gusto.dat.data.min(),
                          pv_gusto.dat.data.max(), 10)
    c2 = tricontourf(pv_gusto, levels=levels2, axes=axes[1])
    fig.colorbar(c2)
    plt.show()

    assert errornorm(pv_true, pv_gusto) < 1e-11
