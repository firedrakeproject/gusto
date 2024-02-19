"""
A moist convective version of the Williamson 2 shallow water test (steady state
geostrophically-balanced flow). The saturation function depends on height,
with a constant background buoyancy/temperature field.
Vapour is initialised very close to saturation and small overshoots will
generate clouds.
"""
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, sin, cos, exp)
import sys

# ----------------------------------------------------------------- #
# Test case parameters
# ----------------------------------------------------------------- #

dt = 120

if '--running-tests' in sys.argv:
    tmax = dt
    dumpfreq = 1
else:
    day = 24*60*60
    tmax = 5*day
    ndumps = 5
    dumpfreq = int(tmax / (ndumps*dt))

R = 6371220.
u_max = 20
phi_0 = 3e4
epsilon = 1/300
theta_0 = epsilon*phi_0**2
g = 9.80616
H = phi_0/g
xi = 0
q0 = 200
beta1 = 110
alpha = 16
gamma_v = 0.98
qprecip = 1e-4
gamma_r = 1e-3

# ----------------------------------------------------------------- #
# Set up model objects
# ----------------------------------------------------------------- #

# Domain
mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=2)
degree = 1
domain = Domain(mesh, dt, 'BDM', degree)
x = SpatialCoordinate(mesh)

# Equations
parameters = ShallowWaterParameters(H=H, g=g)
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R

tracers = [WaterVapour(space='DG'), CloudWater(space='DG'), Rain(space='DG')]

eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                             u_transport_option='vector_advection_form',
                             active_tracers=tracers)

# IO
dirname = "moist_convective_williamson2"
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist_latlon=['D', 'D_error'],
                          dump_nc=True,
                          dump_vtus=True)

diagnostic_fields = [CourantNumber(), RelativeVorticity(),
                     PotentialVorticity(),
                     ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(parameters),
                     ShallowWaterPotentialEnstrophy(),
                     SteadyStateError('u'), SteadyStateError('D'),
                     SteadyStateError('water_vapour'),
                     SteadyStateError('cloud_water')]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)


# define saturation function
def sat_func(x_in):
    h = x_in.split()[1]
    lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])
    numerator = theta_0 + sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))
    denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2
    theta = numerator/denominator
    return q0/(g*h) * exp(20*(theta))


transport_methods = [DGUpwind(eqns, field_name) for field_name in eqns.field_names]

limiter = DG1Limiter(domain.spaces('DG'))

transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D"),
                      SSPRK3(domain, "water_vapour", limiter=limiter),
                      SSPRK3(domain, "cloud_water", limiter=limiter),
                      SSPRK3(domain, "rain", limiter=limiter)
                      ]

linear_solver = MoistConvectiveSWSolver(eqns)

sat_adj = SWSaturationAdjustment(eqns, sat_func,
                                 time_varying_saturation=True,
                                 convective_feedback=True, beta1=beta1,
                                 gamma_v=gamma_v, time_varying_gamma_v=False,
                                 parameters=parameters)

inst_rain = InstantRain(eqns, qprecip, vapour_name="cloud_water",
                        rain_name="rain", gamma_r=gamma_r)

physics_schemes = [(sat_adj, ForwardEuler(domain)),
                   (inst_rain, ForwardEuler(domain))]

stepper = SemiImplicitQuasiNewton(eqns, io,
                                  transport_schemes=transported_fields,
                                  spatial_methods=transport_methods,
                                  linear_solver=linear_solver,
                                  physics_schemes=physics_schemes)

# ----------------------------------------------------------------- #
# Initial conditions
# ----------------------------------------------------------------- #

u0 = stepper.fields("u")
D0 = stepper.fields("D")
v0 = stepper.fields("water_vapour")

lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])

uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, x)
g = parameters.g
w = Omega*R*u_max + (u_max**2)/2
sigma = 0

Dexpr = H - (1/g)*(w)*((sin(phi))**2)
D_for_v = H - (1/g)*(w + sigma)*((sin(phi))**2)

# though this set-up has no buoyancy, we use the expression for theta to set up
# the initial vapour
numerator = theta_0 + sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))
denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2
theta = numerator/denominator

initial_msat = q0/(g*Dexpr) * exp(20*theta)
vexpr = (1 - xi) * initial_msat

u0.project(uexpr)
D0.interpolate(Dexpr)
v0.interpolate(vexpr)

# Set reference profiles
Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
