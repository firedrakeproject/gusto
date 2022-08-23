"""
The Williamson 2 shallow-water test case (solid-body rotation), solved with a
discretisation of the linear shallow-water equations.

This uses an icosahedral mesh of the sphere.
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi
import sys

dt = 3600.
day = 24.*60.*60.
if '--running-tests' in sys.argv:
    tmax = dt
    dumpfreq = 1
else:
    tmax = 5*day
    dumpfreq = int(tmax / (5*dt))

refinements = 3  # number of horizontal cells = 20*(4^refinements)

R = 6371220.
H = 2000.

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements, degree=3)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

output = OutputParameters(dirname='linear_williamson_2',
                          dumpfreq=dumpfreq,
                          steady_state_error_fields=['u', 'D'],
                          log_level='INFO')
parameters = ShallowWaterParameters(H=H)

state = State(mesh,
              dt=dt,
              output=output,
              parameters=parameters)

# Coriolis expression
Omega = parameters.Omega
x = SpatialCoordinate(mesh)
fexpr = 2*Omega*x[2]/R
eqns = LinearShallowWaterEquations(state, "BDM", 1, fexpr=fexpr)

# interpolate initial conditions
# Initial/current conditions
u0 = state.fields("u")
D0 = state.fields("D")
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
u0.project(uexpr)
D0.interpolate(Dexpr)

transport_schemes = [ForwardEuler(state, "D")]

# build time stepper
stepper = CrankNicolson(state, eqns, transport_schemes)

stepper.run(t=0, tmax=tmax)
