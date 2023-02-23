"""
Rossby-Haurwitz wave
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, cos, grad, curl, div, TensorProductElement

# Set up timestepping variables
day = 24. * 60. * 60.
ref = 5
dt_val = [1000, 500, 250]
tmax = 5*day
ndumps = 10

# Shallow Water Parameters
a = 6371220.
H = 5960.

parameters = ShallowWaterParameters(H=H)

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

dt = 250
# Mesh and domain
mesh = IcosahedralSphereMesh(radius=a,
                            refinement_level=ref, degree=1)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)
domain = Domain(mesh, dt, "BDM", 1)

# Equations
lat, lon = latlon_coords(mesh)
Omega = parameters.Omega
fexpr = 2*Omega * x[2] / a
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option='vector_advection_form')

# Output and IO
dirname = 'W6_SIQN_Ref=5_dt=500_diagnostics'
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq,
                        dumplist_latlon=['D', 'D_error'],
                        log_level='INFO')
diagnostic_fields = [CourantNumber(), SteadyStateError('u'), SteadyStateError('D'), RelativeVorticity(), PotentialVorticity(), AbsoluteVorticity(),
                    ShallowWaterKineticEnergy(), ShallowWaterPotentialEnergy(parameters), Divergence(), ShallowWaterPotentialEnstrophy()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport Fields and time stepper
transported_fields = [SSPRK3(domain, "u"),
                    SSPRK3(domain, "D")]

stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields)

# ------------------------------------------------------------------------ #
# Initial Conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')

Vu = domain.spaces("HDiv")
K = Constant(7.847e-6)
w = K
R = 4.
h0 = 8000.
g = parameters.g

# Intilising the velocity field
# CG2 = domain.spaces("CG", 2)
CG2 = FunctionSpace(mesh, 'CG', 2)
psi = Function(CG2)
psiexpr = -a**2 * w * sin(lat) + a**2 * K * cos(lat)**R * sin(lat) * cos(R*lon)
psi.interpolate(psiexpr)
uexpr = domain.perp(grad(psi))
#Dexpr = domain.perp(div(curl(grad(psi))))

# Initilising the depth field
A = (w / 2) * (2 * Omega + w) * cos(lat)**2 + 0.25 * K**2 * cos(lat)**(2 * R) * ((R + 1) * cos(lat)**2 + (2 * R**2 - R - 2) - 2 * R**2 * cos(lat)**(-2))
B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
B = B_frac * cos(lat)**R * ((R**2 + 2 * R + 2) - (R + 1)**2 * cos(lat)**2)
C = (1 / 4) * K**2 * cos(lat)**(2 * R) * ((R + 1)*cos(lat)**2 - (R + 2))
Dexpr = h0 * g + a**2 * (A + B*cos(lon*R) + C * cos(2 * R * lon))

# Finalizing fields and setting reference profiles
u0.project(uexpr)
D0.interpolate(Dexpr / g)
# Dbar is a background field for diagnostics
Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run!
# ------------------------------------------------------------------------ #
stepper.run(t=0, tmax=tmax)
