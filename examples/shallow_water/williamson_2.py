"""
The Williamson 2 shallow-water test case (solid-body rotation), solved with a
discretisation of the non-linear shallow-water equations.

This uses an icosahedral mesh of the sphere, and runs a series of resolutions
to act as a convergence test.
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
    ndumps = 1
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 4000., 4: 2000., 5: 1000., 6: 500.}
    tmax = 5*day
    ndumps = 5

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    dirname = "williamson_2_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    global_normal = x
    mesh.init_cell_orientations(x)

    dumpfreq = int(tmax / (ndumps*dt))
    output = OutputParameters(dirname=dirname,
                              dumpfreq=dumpfreq,
                              dumplist_latlon=['D', 'D_error'],
                              steady_state_error_fields=['D', 'u'],
                              log_level='INFO')

    diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                         ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(),
                         ShallowWaterPotentialEnstrophy()]

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr)

    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    x = SpatialCoordinate(mesh)
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = parameters.g
    Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    transported_fields = [ImplicitMidpoint(state, "u"),
                          SSPRK3(state, "D", subcycles=2)]

    # build time stepper
    stepper = CrankNicolson(state, eqns, transported_fields)

    stepper.run(t=0, tmax=tmax)
