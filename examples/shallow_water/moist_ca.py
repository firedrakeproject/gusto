"""
A moist convective version of the Williamson 2 shallow water test (steady state
geostrophically-balanced flow). The saturation function depends on height,
with a constant background buoyancy/temperature field.
Vapour is initialised very close to saturation and small overshoots will
generate clouds.
"""
from gusto import *
from firedrake import (CubedSphereMesh, SpatialCoordinate, sin, cos, exp)
import sys

# ----------------------------------------------------------------- #
# Test case parameters
# ----------------------------------------------------------------- #

dt = 12000

if '--running-tests' in sys.argv:
    tmax = dt
    dumpfreq = 1
else:
    day = 24*60*60
    tmax = 25*day
    ndumps = 5
    dumpfreq = int(tmax / (ndumps*dt))

R = 6371220.
H = 300

# ----------------------------------------------------------------- #
# Set up model objects
# ----------------------------------------------------------------- #

# Domain
mesh = CubedSphereMesh(radius=R, refinement_level=4, degree=1)
degree = 1
domain = Domain(mesh, dt, 'RTCF', degree)
x = SpatialCoordinate(mesh)

# Equations
parameters = ShallowWaterParameters(H=H)
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R

eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr)

# IO
dirname = "cellular_automaton_forcing"
output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D', 'u_divergence'])

diagnostic_fields = [CourantNumber(),
                     Divergence(),
                     RelativeVorticity(),
                     PotentialVorticity(),
                     ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(parameters),
                     ShallowWaterPotentialEnstrophy()]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

transport_methods = [DGUpwind(eqns, field_name) for field_name in eqns.field_names]

transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D")]
physics_schemes = [(CellularAutomaton(eqns, 10), ForwardEuler(domain))]

stepper = SemiImplicitQuasiNewton(eqns, io,
                                  transport_schemes=transported_fields,
                                  spatial_methods=transport_methods,
                                  physics_schemes=physics_schemes)

# ----------------------------------------------------------------- #
# Initial conditions
# ----------------------------------------------------------------- #

u0 = stepper.fields("u")
D0 = stepper.fields("D")

lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])

dlamda = (30/360)*2*pi
dphi = (10/360)*2*pi
H0 = -10
Dpert = H0 * exp(-(phi/dphi)**2) * exp(-(lamda/dlamda)**2)

Dexpr = H + Dpert

D0.interpolate(Dexpr)

# Set reference profiles
Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
