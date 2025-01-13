from gusto import *
from gusto.rexi import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace,pi

from firedrake.output import VTKFile
import numpy as np

# ----------------------------------------------------------------- #
# Test case parameters
# ----------------------------------------------------------------- #
day = 24.*60.*60.
dt = 200
tmax = 1*dt
ref = 3
# Shallow water parameters
R = 6371220.
H = 10000.
# moist parameters
q0 = 0.0027
nu = 20
beta2 = 9.80616*10
# REXI parameters
h = 0.2
M = 64

parameters = ShallowWaterParameters(H=H, beta2=beta2, nu=nu, q0=q0)

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
eqns = LinearThermalShallowWaterEquations(domain, parameters,
                                          equivalent_buoyancy=True,
                                          fexpr=fexpr)

# ----------------------------------------------------------------- #
# Initial conditions
# ----------------------------------------------------------------- #

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

# prescribed buoyancy
g = Constant(parameters.g)
bexpr = g - cos(phi)

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
    return u_zon*f


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


def initialise_fn(uin, Din):

    uin.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})
    
    # Get coordinates to pass to Dval function
    W = VectorFunctionSpace(mesh, Din.ufl_element())

    X = interpolate(mesh.coordinates, W)
    Din.dat.data[:] = Dval(X.dat.data_ro)
    Din.interpolate(Din - (H/(2*g) * bexpr))

    # Adjust mean value of initial D
    C = Function(Din.function_space()).assign(Constant(1.0))
    area = assemble(C*dx)
    Dmean = assemble(Din*dx)/area
    Din -= Dmean
    Din += Constant(parameters.H)


# initial conditions based on moist thermal version
U_in = Function(eqns.function_space, name="input")
Uexpl = Function(eqns.function_space, name="output")
u0, D0, b_e0, q_t0 = U_in.subfunctions

initialise_fn(u0, D0)

initial_sat = q0*H/(D0) * exp(nu*(1-bexpr/g))
vexpr = 0.98 * initial_sat
b_e_expr = bexpr - beta2*vexpr
b_e0.interpolate(b_e_expr)
q_t0.interpolate(vexpr)
rexi_output = VTKFile("rexi_linear_equivalent_buoyancy_galewsky_jet.pvd")
rexi_output.write(u0, D0, b_e0, q_t0)

# Set reference profiles
Dbar = eqns.X_ref.subfunctions[1]
bebar = eqns.X_ref.subfunctions[2]
qtbar = eqns.X_ref.subfunctions[3]
Dbar.interpolate(H)
qtbar.interpolate(Constant(0.02))
bebar.interpolate(g - beta2*qtbar)

rexi_params = RexiParameters(h=h, M=M)
rexi = Rexi(eqns, rexi_params)

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

rexi.solve(Uexpl, U_in, tmax)
uexpl, Dexpl, b_e_expl, q_t_expl = Uexpl.subfunctions   
u0.assign(uexpl)
D0.assign(Dexpl)
b_e0.assign(b_e_expl)
q_t0.assign(q_t_expl)
rexi_output.write(u0, D0, b_e0, q_t0)

