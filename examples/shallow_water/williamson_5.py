"""
The Williamson 5 shallow-water test case (flow over topography), solved with a
discretisation of the non-linear shallow-water equations.

This uses an icosahedral mesh of the sphere, and runs a series of resolutions.
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, Min)
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
    ndumps = 1
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 900., 4: 450., 5: 225., 6: 112.5}
    tmax = 50*day
    ndumps = 5

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    dirname = "williamson_5_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    dumpfreq = int(tmax / (ndumps*dt))
    output = OutputParameters(dirname=dirname,
                              dumplist_latlon=['D'],
                              dumpfreq=dumpfreq,
                              log_level='INFO')

    diagnostic_fields = [Sum('D', 'topography')]

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    theta, lamda = latlon_coords(mesh)
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = Min(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)
    eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr, bexpr=bexpr)

    # interpolate initial conditions
    u0 = state.fields('u')
    D0 = state.fields('D')
    u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = parameters.g
    Rsq = R**2
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    transported_fields = [ImplicitMidpoint(state, "u"),
                          SSPRK3(state, "D")]

    # build time stepper
    stepper = CrankNicolson(state, eqns, transported_fields)

    stepper.run(t=0, tmax=tmax)
