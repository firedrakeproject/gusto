"""
The Williamson 2 shallow-water test case (solid-body rotation), solved with a
discretisation of the linear shallow-water equations.

This uses an icosahedral mesh of the sphere.
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

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

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements, degree=3)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
parameters = ShallowWaterParameters(H=H)
Omega = parameters.Omega
x = SpatialCoordinate(mesh)
fexpr = 2*Omega*x[2]/R
eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)

# I/O
output = OutputParameters(
    dirname='linear_williamson_2',
    dumpfreq=dumpfreq,
)
diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transport_schemes = [ForwardEuler(domain, "D")]
transport_methods = [DefaultTransport(eqns, "D")]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transport_schemes, transport_methods)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields("u")
D0 = stepper.fields("D")
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
u0.project(uexpr)
D0.interpolate(Dexpr)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
