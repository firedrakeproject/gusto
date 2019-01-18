from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, FunctionSpace)
from math import pi
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 4000., 4: 2000., 5: 1000., 6: 500.}
    tmax = 5*day

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    dirname = "sw_W2_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    global_normal = x
    mesh.init_cell_orientations(x)

    output = OutputParameters(dirname=dirname, dumplist_latlon=['D', 'D_error'], steady_state_error_fields=['D', 'u'])
    diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                         ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(),
                         ShallowWaterPotentialEnstrophy()]

    state = State(mesh, dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    eqns = ShallowWaterEquations(state, family="BDM", degree=1)
    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    x = SpatialCoordinate(mesh)
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    Omega = parameters.Omega
    g = parameters.g
    Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, eqns)))
    advected_fields.append(("D", SSPRK3(state, D0, eqns, subcycles=2)))

    # build time stepper
    stepper = CrankNicolson(state, equations=eqns, advected_fields=advected_fields)

    stepper.run(t=0, tmax=tmax)
