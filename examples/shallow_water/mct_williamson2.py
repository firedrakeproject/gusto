from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, sin, cos, exp)

# ----------------------------------------------------------------- #
# Test case parameters
# ----------------------------------------------------------------- #

day = 24*60*60
tmax = 5*day
ndumps = 5
dt = 100
R = 6371220.
u_max = 20
phi_0 = 3e4
epsilon = 1/300
theta_0 = epsilon*phi_0**2
g = 9.80616
H = phi_0/g
xi = 10e-3
q0 = 135
beta1 = 0.5
beta2 = 0.5
L = 10

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

tracers = [WaterVapour(space='DG'), CloudWater(space='DG')]

eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                             u_transport_option='vector_advection_form',
                             thermal=True, active_tracers=tracers)

# IO
dirname = "mct_williamson2"
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist_latlon=['D', 'D_error'],
                          log_level='INFO')

diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                     ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(parameters),
                     ShallowWaterPotentialEnstrophy(),
                     SteadyStateError('u'), SteadyStateError('D'),
                     SteadyStateError('b'), SteadyStateError('water_vapour'),
                     SteadyStateError('cloud_water')]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)


# Physics schemes
# Saturation function
def sat_func(x_in):
    h = x_in.split()[1]
    b = x_in.split()[2]
    return q0/(g*h) * exp(20*(1 - b/g))


# Feedback proportionality is dependent on h and b
def gamma_v(x_in):
    h = x_in.split()[1]
    b = x_in.split()[2]
    return (1 + 10*(20*q0/g*h * exp(20*(1 - b/g))))**(-1)

ReversibleAdjustment(eqns, sat_func, L, time_varying_saturation=True,
                     parameters=parameters,
                     convective_feedback=True, beta1=beta1,
                     thermal_feedback=True, beta2=beta2,
                     gamma_v=gamma_v, time_varying_gamma_v=True)

# Time stepper
stepper = Timestepper(eqns, RK4(domain), io)

# ----------------------------------------------------------------- #
# Initial conditions
# ----------------------------------------------------------------- #

u0 = stepper.fields("u")
D0 = stepper.fields("D")
b0 = stepper.fields("b")
v0 = stepper.fields("water_vapour")

phi, lamda = latlon_coords(mesh)

uexpr = sphere_to_cartesian(mesh, u_max*cos(phi), 0)
g = parameters.g
w = Omega*R*u_max + (u_max**2)/2
sigma = w/10

Dexpr = H - (1/g)*(w + sigma)*((sin(phi))**2)

numerator = theta_0 + sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))

denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2

theta = numerator/denominator

bexpr = parameters.g * (1 - theta)

initial_msat = q0/(g*Dexpr) * exp(20*theta)
vexpr = (1 - xi) * initial_msat

u0.project(uexpr)
D0.interpolate(Dexpr)
b0.interpolate(bexpr)
v0.interpolate(vexpr)

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
