from gusto import *
from firedrake import IcosahedralSphereMesh, \
    Constant, ge, le, exp, cos, \
    conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, pi, CellNormal

import numpy as np
import sys

day = 24.*60.*60.
dt = 240.

if '--running-tests' in sys.argv:
    ref_level = 3
    dt = 480.
    tmax = 1440.
    cubed_sphere = False
else:
    # setup resolution and timestepping parameters
    cubed_sphere = True
    if cubed_sphere:
        ncells = 8  # 24
        dt = 200.
    else:
        ref_level = 4
        dt = 240.
    tmax = 6*day

# setup shallow water parameters
radius = 6371220.
H = 10000.

perturb = True
if perturb:
    dirname = "mm_ot_sw_galewsky_jet_perturbed_dt%s" % dt
else:
    dirname = "mm_ot_sw_galewsky_jet_unperturbed"

# Domain
if cubed_sphere:
    dirname += "_cs%s" % ncells
    mesh = GeneralCubedSphereMesh(radius=radius,
                                  num_cells_per_edge_of_panel=ncells,
                                  degree=2)
    family = "RTCF"
else:
    dirname += "_is%s" % ref_level
    mesh = IcosahedralSphereMesh(radius=radius,
                                 refinement_level=ref_level, degree=2)
    family = "BDM"
domain = Domain(mesh, dt, family, 1, move_mesh=True)

# Equation
parameters = ShallowWaterParameters(mesh, H=H)
eqns = ShallowWaterEquations(domain, parameters,
                             u_transport_option="vector_invariant_form")

# I/O
output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D', 'PotentialVorticity',
                                           'RelativeVorticity'])

pv = PotentialVorticity()
diagnostic_fields = [pv, RelativeVorticity()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D")]

transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]


# Mesh movement
def update_pv():
    pv()


def reinterpolate_coriolis():
    domain.k = interpolate(x/radius, domain.mesh.coordinates.function_space())
    domain.outward_normals.interpolate(CellNormal(domain.mesh))
    eqns.prescribed_fields("coriolis").interpolate(fexpr)


monitor = MonitorFunction("PotentialVorticity", adapt_to="gradient")
mesh_generator = OptimalTransportMeshGenerator(domain.mesh,
                                               monitor,
                                               pre_meshgen_callback=update_pv,
                                               post_meshgen_callback=reinterpolate_coriolis)

# Time stepper
stepper = MeshMovement(eqns, io, transported_fields,
                       spatial_methods=transport_methods,
                       mesh_generator=mesh_generator)

# initial conditions
u0_field = stepper.fields("u")
D0_field = stepper.fields("D")

# Parameters
umax = 80.0          # amplitude of jet wind speed, in m/s
phi0 = pi/7          # lower latitude of initial jet, in rad
phi1 = pi/2 - phi0   # upper latitude of initial jet, in rad
phi2 = pi/4          # central latitude of perturbation to jet, in rad
alpha = 1.0/3        # zonal width parameter of perturbation, in rad
beta = 1.0/15        # meridional width parameter of perturbation, in rad
h_hat = 120.0        # strength of perturbation, in m
g = parameters.g
Omega = parameters.Omega
e_n = np.exp(-4./((phi1-phi0)**2))

xyz = SpatialCoordinate(mesh)
lon, lat, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
lat_VD = Function(D0_field.function_space()).interpolate(lat)

# ------------------------------------------------------------------------ #
# Obtain u and D (by integration of analytic expression)
# ------------------------------------------------------------------------ #

# Wind -- UFL expression
u_zonal = conditional(
    lat <= phi0, 0.0,
    conditional(
        lat >= phi1, 0.0,
        umax / e_n * exp(1.0 / ((lat - phi0) * (lat - phi1)))
    )
)
uexpr = xyz_vector_from_lonlatr(u_zonal, Constant(0.0), Constant(0.0), xyz)

# Numpy function
def u_func(y):
    u_array = np.where(
        y <= phi0, 0.0,
        np.where(
            y >= phi1, 0.0,
            umax / e_n * np.exp(1.0 / ((y - phi0) * (y - phi1)))
        )
    )
    return u_array

# Function for depth field in terms of u function
def h_func(y):
    h_array = u_func(y)*float(radius)/float(g)*(
        2*float(Omega)*np.sin(y)
        + u_func(y)*np.tan(y)/float(radius)
    )

    return h_array

# Find h from numerical integral
D0_integral = Function(D0_field.function_space())
h_integral = NumericalIntegral(-pi/2, pi/2)
h_integral.tabulate(h_func)
D0_integral.dat.data[:] = h_integral.evaluate_at(lat_VD.dat.data[:])
Dexpr = H - D0_integral

# Obtain fields
u0_field.project(uexpr)
D0_field.interpolate(Dexpr)

# Adjust mean value of initial D
C = Function(D0_field.function_space()).assign(Constant(1.0))
area = assemble(C*dx)
Dmean = assemble(D0_field*dx)/area
D0_field -= Dmean
D0_field += Constant(H)

# Background field, store in object for use in diagnostics
Dbar = Function(D0_field.function_space()).assign(D0_field)

# ------------------------------------------------------------------------ #
# Apply perturbation
# ------------------------------------------------------------------------ #

h_pert = h_hat*cos(lat)*exp(-(lon/alpha)**2)*exp(-((phi2-lat)/beta)**2)
D0_field.interpolate(Dexpr + h_pert)

def initialise_fn():
    u0_field = stepper.fields("u")
    D0_field = stepper.fields("D")

    u0_field.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})

    # Find h from numerical integral
    D0_integral = Function(D0_field.function_space())
    h_integral = NumericalIntegral(-pi/2, pi/2)
    h_integral.tabulate(h_func)
    D0_integral.dat.data[:] = h_integral.evaluate_at(lat_VD.dat.data[:])
    Dexpr = H - D0_integral
    D0_field.interpolate(Dexpr)
    C = Function(D0_field.function_space()).assign(Constant(1.0))
    area = assemble(C*dx)
    Dmean = assemble(D0_field*dx)/area
    D0_field -= Dmean
    D0_field += Constant(H)

    if perturb:
        h_pert = h_hat*cos(lat)*exp(-(lon/alpha)**2)*exp(-((phi2-lat)/beta)**2)
        D0_field.interpolate(D0_field + h_pert)
    domain.k = interpolate(xyz/radius, domain.mesh.coordinates.function_space())
    domain.outward_normals.interpolate(CellNormal(domain.mesh))
    #eqns.prescribed_fields("coriolis").interpolate(fexpr)
    pv()

pv.setup(domain, stepper.fields)
mesh_generator.get_first_mesh(initialise_fn)

domain.k = interpolate(xyz/radius, domain.mesh.coordinates.function_space())
domain.outward_normals.interpolate(CellNormal(domain.mesh))
eqns.prescribed_fields("coriolis").interpolate(fexpr)
pv()

Dbar = Function(D0_field.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

stepper.run(t=0, tmax=tmax)









# get lat lon coordinates
x, y, z = SpatialCoordinate(mesh)
theta, lamda, _ = lonlatr_from_xyz(x, y, z)

# expressions for meridional and zonal velocity
u_max = 80.0
theta0 = pi/7.
theta1 = pi/2. - theta0
en = np.exp(-4./((theta1-theta0)**2))
u_zonal_expr = (u_max/en)*exp(1/((theta - theta0)*(theta - theta1)))
u_zonal = conditional(ge(theta, theta0), conditional(le(theta, theta1), u_zonal_expr, 0.), 0.)
u_merid = 0.0

# get cartesian components of velocity
Rc = Constant(R)
uexpr = xyz_vector_from_lonlatr(u_zonal, u_merid, Rc, (x, y, z))
u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})

g = Constant(parameters.g)


def D_integrand(th):
    # Initial D field is calculated by integrating D_integrand w.r.t. theta
    # Assumes the input is between theta0 and theta1.
    # Note that this function operates on vectorized input.
    from numpy import exp, sin, tan
    f = 2.0*parameters.Omega*sin(th)
    u_zon = (80.0/en)*exp(1.0/((th - theta0)*(th - theta1)))
    return u_zon*(f + tan(th)*u_zon/R)


def Dval(X):
    # Function to return value of D at X
    from scipy import integrate

    # Preallocate output array
    val = np.zeros(len(X))

    angles = np.zeros(len(X))

    # Minimize work by only calculating integrals for points with
    # theta between theta_0 and theta_1.
    # For theta <= theta_0, the integral is 0
    # For theta >= theta_1, the integral is constant.

    # Precalculate this constant:
    poledepth, _ = integrate.fixed_quad(D_integrand, theta0, theta1, n=64)
    poledepth *= -R/parameters.g

    angles[:] = np.arcsin(X[:, 2]/R)

    for ii in range(len(X)):
        if angles[ii] <= theta0:
            val[ii] = 0.0
        elif angles[ii] >= theta1:
            val[ii] = poledepth
        else:
            # Fixed quadrature with 64 points gives absolute errors below 1e-13
            # for a quantity of order 1e-3.
            v, _ = integrate.fixed_quad(D_integrand, theta0, angles[ii], n=64)
            print(R, parameters.g, v)
            val[ii] = -(R/parameters.g)*v

    return val


# Get coordinates to pass to Dval function
W = VectorFunctionSpace(mesh, D0.ufl_element())
X = Function(W).interpolate(mesh.coordinates)
D0.dat.data[:] = Dval(X.dat.data_ro)

# Adjust mean value of initial D
C = Function(D0.function_space()).assign(Constant(1.0))
area = assemble(C*dx)
Dmean = assemble(D0*dx)/area
D0 -= Dmean
D0 += Constant(parameters.H)

# optional perturbation
if perturb:
    alpha = Constant(1/3.)
    beta = Constant(1/15.)
    Dhat = Constant(120.)
    theta2 = Constant(pi/4.)
    g = Constant(parameters.g)
    D_pert = Function(D0.function_space()).interpolate(Dhat*cos(theta)*exp(-(lamda/alpha)**2)*exp(-((theta2 - theta)/beta)**2))
    D0 += D_pert


def initialise_fn():
    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})

    X = interpolate(domain.mesh.coordinates, W)
    D0.dat.data[:] = Dval(X.dat.data_ro)
    area = assemble(C*dx)
    Dmean = assemble(D0*dx)/area
    D0 -= Dmean
    D0 += parameters.H
    if perturb:
        theta, lamda = latlon_coords(domain.mesh)
        D_pert.interpolate(Dhat*cos(theta)*exp(-(lamda/alpha)**2)*exp(-((theta2 - theta)/beta)**2))
        D0 += D_pert
    domain.k = interpolate(x/R, domain.mesh.coordinates.function_space())
    domain.outward_normals.interpolate(CellNormal(domain.mesh))
    eqns.prescribed_fields("coriolis").interpolate(fexpr)
    pv()


pv.setup(domain, stepper.fields)
mesh_generator.get_first_mesh(initialise_fn)

domain.k = interpolate(x/R, domain.mesh.coordinates.function_space())
domain.outward_normals.interpolate(CellNormal(domain.mesh))
eqns.prescribed_fields("coriolis").interpolate(fexpr)
pv()

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

stepper.run(t=0, tmax=tmax)
