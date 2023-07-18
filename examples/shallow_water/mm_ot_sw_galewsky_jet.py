from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace, pi, CellNormal

import numpy as np

day = 24.*60.*60.
dt = 240.
tmax = 6*day
ref_level = 4

# setup shallow water parameters
R = 6371220.
H = 10000.
parameters = ShallowWaterParameters(H=H)

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=2)
domain = Domain(mesh, dt, 'BDM', 1, move_mesh=True)

# Equation
Omega = parameters.Omega
x = SpatialCoordinate(domain.mesh)
fexpr = 2*Omega*x[2]/R
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option="circulation_form")

# I/O
perturb = True
if perturb:
    dirname = "mm_ot_sw_galewsky_jet_perturbed_ref%s_dt%s" % (ref_level, dt)
else:
    dirname = "mm_ot_sw_galewsky_jet_unperturbed"

output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D', 'PotentialVorticity',
                                           'RelativeVorticity'],
                          log_level="INFO")
pv = PotentialVorticity()
diagnostic_fields = [pv, RelativeVorticity()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [ImplicitMidpoint(domain, "u"),
                      SSPRK3(domain, "D")]


# Mesh movement

def update_pv():
    pv()

def reinterpolate_coriolis():
    domain.k = interpolate(x/R, domain.mesh.coordinates.function_space())
    domain.outward_normals.interpolate(CellNormal(domain.mesh))
    eqns.prescribed_fields("coriolis").interpolate(fexpr)

monitor = MonitorFunction("PotentialVorticity", adapt_to="gradient")
mesh_generator = OptimalTransportMeshGenerator(domain.mesh,
                                               monitor,
                                               pre_meshgen_callback=update_pv,
                                               post_meshgen_callback=reinterpolate_coriolis)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  mesh_generator=mesh_generator)
#stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields)

# initial conditions
u0 = stepper.fields("u")
D0 = stepper.fields("D")

# get lat lon coordinates
theta, lamda = latlon_coords(mesh)

# expressions for meridional and zonal velocity
u_max = 80.0
theta0 = pi/7.
theta1 = pi/2. - theta0
en = np.exp(-4./((theta1-theta0)**2))
u_zonal_expr = (u_max/en)*exp(1/((theta - theta0)*(theta - theta1)))
u_zonal = conditional(ge(theta, theta0), conditional(le(theta, theta1), u_zonal_expr, 0.), 0.)
u_merid = 0.0

# get cartesian components of velocity
uexpr = sphere_to_cartesian(mesh, u_zonal, u_merid)
u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})

Rc = Constant(R)
g = Constant(parameters.g)


def D_integrand(th):
    # Initial D field is calculated by integrating D_integrand w.r.t. theta
    # Assumes the input is between theta0 and theta1.
    # Note that this function operates on vectorized input.
    from scipy import exp, sin, tan
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
            val[ii] = -(R/parameters.g)*v

    return val


# Get coordinates to pass to Dval function
W = VectorFunctionSpace(mesh, D0.ufl_element())
X = interpolate(mesh.coordinates, W)
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

# stepper.prescribed_uexpr = sphere_to_cartesian(mesh, u_zonal, u_merid)
pv.setup(domain, stepper.fields)
mesh_generator.get_first_mesh(initialise_fn)

domain.k = interpolate(x/R, domain.mesh.coordinates.function_space())
domain.outward_normals.interpolate(CellNormal(domain.mesh))
eqns.prescribed_fields("coriolis").interpolate(fexpr)
pv()

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

stepper.run(t=0, tmax=tmax)
