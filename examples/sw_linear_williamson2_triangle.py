from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector)
from math import pi
import sys

dt = 3600.
day = 24.*60.*60.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 5*day

refinements = 3  # number of horizontal cells = 20*(4^refinements)

R = 6371220.
H = 2000.

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements, degree=3)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

fieldlist = ['u', 'D']
output = OutputParameters(dirname='sw_linear_w2', steady_state_error_fields=['u', 'D'])
parameters = ShallowWaterParameters(H=H)

state = State(mesh, dt=dt,
              output=output,
              parameters=parameters)

eqns = LinearShallowWaterEquations(state, family="BDM", degree=1)

# interpolate initial conditions
# Initial/current conditions
u0 = state.fields("u")
D0 = state.fields("D")
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
Omega = parameters.Omega
g = parameters.g
Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
u0.project(uexpr)
D0.interpolate(Dexpr)
state.initialise([('u', u0),
                  ('D', D0)])

advected_fields = []
advected_fields.append(("D", ForwardEuler(state, D0, eqns)))

# build time stepper
stepper = CrankNicolson(state, equations=eqns, advected_fields=advected_fields)

stepper.run(t=0, tmax=tmax)
