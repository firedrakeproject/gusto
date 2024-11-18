from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, pi

import numpy as np

# ----------------------------------------------------------------- #
# Test case parameters
# ----------------------------------------------------------------- #
day = 24.*60.*60.
tmax = 6*day
dt = 2000
ndumps = 24
ref = 3
# Shallow water parameters
R = 6371220.
H = 10000.
parameters = ShallowWaterParameters(H=H)

# ----------------------------------------------------------------- #
# Set up model objects
# ----------------------------------------------------------------- #

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref, degree=2)
x = SpatialCoordinate(mesh)

domain = Domain(mesh, dt, 'BDM', 1)

# Equation
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
eqns = LinearThermalShallowWaterEquations(domain, parameters, fexpr=fexpr)

dirname = "linear_thermal_galewsky_jet"
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                          dumpfreq=1,
                          dumplist_latlon=['D', 'b',
                                           'PotentialVorticity',
                                           'RelativeVorticity'],
                          dump_nc=True,
                          dump_vtus=True,
                          chkptfreq=1)

diagnostic_fields = [PotentialVorticity(), RelativeVorticity(), CourantNumber()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

transport_methods = [DefaultTransport(eqns, "D")]

stepper = Timestepper(eqns, RK4(domain), io, spatial_methods=transport_methods)

# ----------------------------------------------------------------- #
# Initial conditions
# ----------------------------------------------------------------- #

u0 = stepper.fields("u")
D0 = stepper.fields("D")
b0 = stepper.fields("b")

# get lat lon coordinates
lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])

# expressions for meridional and zonal velocity
u_max = 80.0
phi0 = pi/7.
phi1 = pi/2. - phi0
en = np.exp(-4./((phi1-phi0)**2))
u_zonal_expr = (u_max/en)*exp(1/((phi - phi0)*(phi - phi1)))
u_zonal = conditional(ge(phi, phi0), conditional(le(phi, phi1), u_zonal_expr, 0.), 0.)
u_merid = 0.0

# get cartesian components of velocity
uexpr = xyz_vector_from_lonlatr(u_zonal, 0, 0, x)

# expression for buoyancy
g = Constant(parameters.g)
bexpr = g - cos(phi)
b0.interpolate(bexpr)

# ----------------------------------------------------------------------- #
# Compute balanced initial depth - this code based on the dry Galewsky jet
# ----------------------------------------------------------------------- #


def D_integrand(th):
    # Initial D field is calculated by integrating D_integrand w.r.t. phi
    # Assumes the input is between phi0 and phi1.
    # Note that this function operates on vectorized input.
    from numpy import exp, sin, tan
    f = 2.0*parameters.Omega*sin(th)
    u_zon = (80.0/en)*exp(1.0/((th - phi0)*(th - phi1)))
    return u_zon*(f + tan(th)*u_zon/R)


def Dval(X):
    # Function to return value of D at X
    from scipy import integrate

    # Preallocate output array
    val = np.zeros(len(X))

    angles = np.zeros(len(X))

    # Minimize work by only calculating integrals for points with
    # phi between phi_0 and phi_1.
    # For phi <= phi_0, the integral is 0
    # For phi >= phi_1, the integral is constant.

    # Precalculate this constant:
    poledepth, _ = integrate.fixed_quad(D_integrand, phi0, phi1, n=64)
    poledepth *= -R/parameters.g

    angles[:] = np.arcsin(X[:, 2]/R)

    for ii in range(len(X)):
        if angles[ii] <= phi0:
            val[ii] = 0.0
        elif angles[ii] >= phi1:
            val[ii] = poledepth
        else:
            # Fixed quadrature with 64 points gives absolute errors below 1e-13
            # for a quantity of order 1e-3.
            v, _ = integrate.fixed_quad(D_integrand, phi0, angles[ii], n=64)
            val[ii] = -(R/parameters.g)*v

    return val


def initialise_fn():
    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})

    # Get coordinates to pass to Dval function
    W = VectorFunctionSpace(mesh, D0.ufl_element())

    X = interpolate(mesh.coordinates, W)
    D0.dat.data[:] = Dval(X.dat.data_ro)
    D0.interpolate(D0 - (H/(2*g) * b0))

    # Adjust mean value of initial D
    C = Function(D0.function_space()).assign(Constant(1.0))
    area = assemble(C*dx)
    Dmean = assemble(D0*dx)/area
    D0 -= Dmean
    D0 += Constant(parameters.H)


initialise_fn()

# Set reference profiles
Dbar = Function(D0.function_space()).assign(H)
bbar = Function(b0.function_space()).interpolate(g)
stepper.set_reference_profiles([('D', Dbar), ('b', bbar)])

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

stepper.run(t=0, tmax=20*dt)
