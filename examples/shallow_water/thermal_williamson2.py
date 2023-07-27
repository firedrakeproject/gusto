from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, sin, cos,
                       as_vector)

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

# ----------------------------------------------------------------- #
# Set up model objects
# ----------------------------------------------------------------- #

# Domain
mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=2)
degree = 1
domain = Domain(mesh, dt, 'BDM', degree)
x = SpatialCoordinate(mesh)

# Equations
params = ShallowWaterParameters(H=H, g=g)
Omega = params.Omega
fexpr = 2*Omega*x[2]/R
eqns = ShallowWaterEquations(domain, params, fexpr=fexpr, u_transport_option='vector_advection_form', thermal=True)

# IO
dirname = "thermal_williamson2"
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist_latlon=['D', 'D_error'],
                          log_level='INFO')

diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                     ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(params),
                     ShallowWaterPotentialEnstrophy(),
                     SteadyStateError('u'), SteadyStateError('D')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)
transport_methods = [DGUpwind(eqns, "u"),
                     DGUpwind(eqns, "D"),
                     DGUpwind(eqns, "b")]

# Time stepper
stepper = Timestepper(eqns, RK4(domain), io, spatial_methods=transport_methods)

# ----------------------------------------------------------------- #
# Initial conditions
# ----------------------------------------------------------------- #

u0 = stepper.fields("u")
D0 = stepper.fields("D")
b0 = stepper.fields("b")

lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])

uexpr = lonlatr_vector_from_xyz(as_vector([u_max*cos(phi), 0, 0]), x)
g = params.g
w = Omega*R*u_max + (u_max**2)/2
sigma = 0

Dexpr = H - (1/g)*(w + sigma)*((sin(phi))**2)

numerator = theta_0 - sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))

denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2

theta = numerator/denominator

bexpr = params.g * (1 - theta)

u0.project(uexpr)
D0.interpolate(Dexpr)
b0.interpolate(bexpr)

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
