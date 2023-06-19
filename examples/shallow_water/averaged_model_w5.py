"""
The Williamson 5 shallow-water test case (flow over topography), solved with a
discretisation of the non-linear shallow-water equations.

This uses an icosahedral mesh of the sphere, and runs a series of resolutions.
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value)
from math import ceil

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.
ref_level = 3
dt = 0.5*60*60
tmax = dt

# setup shallow water parameters
R = 6371220.
H = 5960.
g = 9.8

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(g=g, H=H)

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
theta, lamda = latlon_coords(mesh)
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

# I/O
dirname = "avg_williamson_5_ref%s_dt%s" % (ref_level, dt)
dumpfreq = 1
output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D'],
                          dumpfreq=dumpfreq,
                          log_level='INFO')
diagnostic_fields = [Sum('D', 'topography')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)


eigs = [0.003465, 0.007274, 0.014955]
eta = 1
L = eigs[ref_level-3]*dt*eta
ppp = 3
Mbar = ceil(ppp*eta*dt*eigs[ref_level-3]/2/pi)
print(Mbar)
Mbar = 3
print(Mbar)

ncheb = 10000
tol = 1e-6
filter_val = 0.75

exp_method = Cheby(eqns, ncheb, tol, L, filter_val)
exp_method2 = Cheby(eqns, ncheb, tol, L)
scheme = Heun(domain, subcycles=2)

avg_model = AveragedModel(eta, Mbar, exp_method, exp_method2, scheme)

# Time stepper
stepper = Timestepper(eqns, avg_model, io)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
print(g, Omega, H, R)
Dexpr = - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g

u0.project(uexpr)
D0.interpolate(Dexpr)

Dbar = Function(D0.function_space()).assign(0)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
