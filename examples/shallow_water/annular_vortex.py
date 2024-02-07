"""
Set up Martian annular vortex experiment!
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, sin, cos)

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.

### max runtime currently 1 day
tmax = day
### timestep
dt = 450.

# setup shallow water parameters
#R = 6371220.    ### radius of Earth - change for Mars!
R = 3389500.    # Mars value (3389500)
#H = 5960.      ### probably this changes too!
H = 17000.      # Will's Mars value

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
#Omega = parameters.Omega
Omega = 2*pi/88774
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
#bexpr = 0.
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr)

# I/O (input/output)
dirname = "annular_vortex_mars_no_u"
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
#u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
u_max = 0.1
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = 3.71
#g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr
#Dexpr = 0.

u0.project(uexpr)
D0.interpolate(Dexpr)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)


# this bit is solving for the initial conditions

# need to setup totintegral to be an integral in latitude from -pi/2 to +pi/2
# and cumulintegral to be an integral in latitude from -pi/2 to current latitude

twomega = 2 * Omega
twomegarad = twomega * radius

alpha = 0.5

hbart = 1
phibar = H

lat = theta
sinlat = sin(lat)
coslat = cos(lat)

# need to find the correct weighting function for the integrals as lat not necessarily equally spaced
da = coslat * pi



f = twomega * sinlat

# not sure if this works in gusto?
hn = [1] * len(lat)

rlat1 = pi / 180. * 45.
rlat2 = pi / 180. * 50.
qp = twomega / hbart
qt0 = twomega * sinlat / hbart
qt = qt0


# not really sure what this conditional bit is meant to be doing
qt = condit(lat > 0., 0.3*qp,
        condit(lat > rlat1, 1.6*qp,
            condit(lat > rlat2, qp, qt0)))

# iteration loop
k=0
while k<50:
    ctop = totintegral(qt*hn*da)
    cbot = totintegral(hn*da)

    coff = -ctop/cbot

    if k >= 1:
        zn0 = zn
    zn = (coff + qt) * hn - f
    if k >= 1:
        zn = alpha * zn + (1 - alpha) * zn0
    
    un = -cumulintegral(zn * da) * R / coslat

    dhdmu = -(un / coslat + twomegarad) * un * sinlat / (coslat * phibar)

    hn = cumulintegral(dhdmu * da)

    havgn = totintegral(hn * da)/totintegral(da)
    hn = hn - havgn + hbart

    k += 1

    qn = (f + zn)/hn -coff
    error = totintegral(sqrt((qn - qt)**2))