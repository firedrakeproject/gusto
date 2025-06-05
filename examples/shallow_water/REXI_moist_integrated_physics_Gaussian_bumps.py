"""
Moist thermal integrated physics version of the linear shallow water test with 
three Gaussian bumps from Schreiber and Loft (2018). There is one Guassian in
the height and the two others in the buoyancy field. The idea is that all moisture remains
as vapour throughout the test.
This needs to be run with a hack to deal with the terms that don't have trial
functions in them. This is implemented in this commit:
https://github.com/firedrakeproject/gusto/commit/6a3acc44df453f43c70e9cdc5cfb302e3fc25f29#diff-6156a5fc9f1bd676f88eb7407df89fdcaee05ac6ef7538b45fe38bb1528ee054
It is also necessary to include the hack to make all moisture vapour, commited here:
https://github.com/firedrakeproject/gusto/commit/76166e0126e7b65cee90476dcfb24a6dbf58a6d8
(The branch these are implemented on is too old to work so just copy these two commits.)
"""
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       Function, acos, cos, sin, pi, exp, Constant,
                       functionspaceimpl, COMM_WORLD, Ensemble)
from firedrake.output import VTKFile


# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 300
day = 24*60*60
hour = 60*60
# tmax = 1.5*day
tmax = 7500
ref = 3

# shallow water parameters
R = 6371220.
H = 10000.

# moist parameters
q0 = 0.03
nu = 1.5
beta2 = 9.80616*10

# REXI parameters (need to be specified as Constants)
# h = Constant(0.15)
# M = Constant(4098)
h = Constant(1)
#M = Constant(450)
M = Constant(26)

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #
parallel = False

# Domain
if parallel:
    ensemble = Ensemble(COMM_WORLD, 1)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref, degree=2, comm=ensemble.comm)
else:
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref, degree=2)
xyz = SpatialCoordinate(mesh)
#mesh.init_cell_orientations(xyz)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
parameters = ShallowWaterParameters(mesh, H=H, q0=q0, nu=nu, beta2=beta2)
Omega = parameters.Omega
fexpr = 2*Omega*xyz[2]/R
eqns = LinearThermalShallowWaterEquations(domain, parameters,
                                          equivalent_buoyancy=True,
                                          fexpr=fexpr)
#for t in eqns.residual:
#    print(t.has_label(constant_label))

# estimate core count for Pileus
print(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} ')

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

def d(lamda1, phi1, lamda2, phi2):
    return acos(sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(lamda1-lamda2))


def psi_d(lamda, phi, lamda_c, phi_c, p):
    return exp(-d(lamda_c, phi_c, lamda, phi)**(2)*p) * 0.1*H


def psi_b(lamda, phi, lamda_c, phi_c, p):
    return exp(-d(lamda_c, phi_c, lamda, phi)**(2)*p) * 0.1*parameters.g


lamda, phi, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

psi1 = psi_d(lamda, phi, 0.2*pi, pi/3, 20)
psi2 = psi_b(lamda, phi, 1.2*pi, pi/5, 80)
psi3 = psi_b(lamda, phi, 1.6*pi, -pi/4, 360)

Dexpr = H + psi1
bexpr = parameters.g + cos(phi) + psi2 + psi3

# initial saturation (vapour is set below this)
sat_expr = q0*H/Dexpr * exp(nu*(1 - bexpr/parameters.g))
vexpr = 0.5 * sat_expr

U_in = Function(eqns.function_space, name="input")
Uexpl = Function(eqns.function_space, name="output")
u0, D0, b_e0, q_t0 = U_in.subfunctions

D0.interpolate(Dexpr)
b_e0.interpolate(bexpr)
q_t0.interpolate(vexpr)

rexi_output = VTKFile("rexi_moist_integrated_physics_gaussians.pvd")

# Set reference profiles
Dbar = eqns.X_ref.subfunctions[1]
Dbar.assign(H)
bebar = eqns.X_ref.subfunctions[2]
bebar.interpolate(parameters.g-cos(phi))
qtbar = eqns.X_ref.subfunctions[3]
qtbar.interpolate(vexpr)

rexi_params = RexiParameters(M=M, h=h)
if parallel:
    rexi = Rexi(eqns, rexi_params, manager=ensemble, compute_eigenvalues=True)
else:
    rexi = Rexi(eqns, rexi_params, compute_eigenvalues=True)

# output initial conditions
rexi_output.write(u0, D0, b_e0, q_t0)

# output to a lat-lon mesh
mesh_ll = get_flat_latlon_mesh(mesh)
ll_D0 = Function(functionspaceimpl.WithGeometry.create(
                        D0.function_space(), mesh_ll),
                    val=D0.topological)
ll_b_e0 = Function(functionspaceimpl.WithGeometry.create(
                        b_e0.function_space(), mesh_ll),
                    val=b_e0.topological)
ll_q_t0 = Function(functionspaceimpl.WithGeometry.create(
                        q_t0.function_space(), mesh_ll),
                    val=q_t0.topological)
outfile_ll = VTKFile("latlon_rexi_moist_integrated_physics_gaussians.pvd")
outfile_ll.write(ll_D0, ll_b_e0, ll_q_t0)

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #
rexi.X_ref.assign(eqns.X_ref)
rexi.solve(Uexpl, U_in, tmax)
uexpl, Dexpl, beexpl, qtexpl = Uexpl.subfunctions   
u0.assign(uexpl)
D0.assign(Dexpl)
b_e0.assign(beexpl)
q_t0.assign(qtexpl)
rexi_output.write(u0, D0, b_e0, q_t0)
outfile_ll.write(ll_D0, ll_b_e0, ll_q_t0)
