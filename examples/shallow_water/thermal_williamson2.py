from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi,
                       cos, sin, Constant)

day = 24*60*60
tmax = 5*day
ndumps = 5
dt = 100 #3000

# setup shallow water parameters
R = 6371220.
H = 5960.
Omega = 7.292e-5

parameters = ShallowWaterParameters(H=H, Omega=Omega)

dirname = "thermal_williamson2"

mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=1)

x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

phi, lamda = latlon_coords(mesh)

dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                          dumpfreq=1,
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
print(Omega)
fexpr = 2*Omega*x[2]/R
eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr, thermal=True)

# initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
b0 = state.fields("b")

u_max = 20
uexpr = sphere_to_cartesian(mesh, u_max*cos(phi), 0)
g = parameters.g
w = Omega*R*u_max + (u_max**2)/2
sigma = w/10

Dexpr = H - (1/g)*(w + sigma)*((sin(phi))**2)

phi_0 = 3e4
epsilon = 0.1
theta_0 = epsilon*phi_0**2

# theta = (theta_0 - sigma*((
#     cos(phi))**2) * ((w + sigma)(cos(phi))**2 + 2*(phi_0 - w - sigma)))/(
#         phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*
#         (sin(phi))**2)

numerator = theta_0 - sigma*((cos(phi))**2) * ((w + sigma)(cos(phi))**2 + 2*(phi_0 - w - sigma))

denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2

theta = numerator/denominator

bexpr = parameters.g * (1 - theta)

u0.project(uexpr)
D0.interpolate(Dexpr)
b0.interpolate(bexpr)

# build time stepper
stepper = Timestepper(eqns, RK4(state), state)

stepper.run(t=0, tmax=5*dt)
