"""
Set up Martian annular vortex experiment!
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value)

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.

### max runtime currently 1 day
tmax = day
### timestep
dt = 450.

# setup shallow water parameters
R = 6371220.    ### radius of Earth - change for Mars!
H = 5960.       ### probably this changes too!

### setup shallow water parameters - can also change g and Omega as required
parameters = ShallowWaterParameters(H=H)

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=4, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation, including mountain given by bexpr
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
lsq = (lamda - lamda_c)**2
theta_c = pi/6.
thsq = (theta - theta_c)**2
rsq = min_value(R0sq, lsq+thsq)
r = sqrt(rsq)
bexpr = 2000 * (1 - r/R0)
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr)

# I/O (input/output)
dirname = "annular_vortex"
output = OutputParameters(dirname=dirname)
diagnostic_fields = [Sum('D', 'topography')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D")]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)

# ------------------------------------------------------------------------ #
# Initial conditions - these need changing!
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

u0.project(uexpr)
D0.interpolate(Dexpr)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
