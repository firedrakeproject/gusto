"""
The 1 metre high mountain test case. This is solved with the non-hydrostatic
compressible Euler equations.
"""

from gusto import *
from firedrake import (as_vector, VectorFunctionSpace,
                       PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       exp, pi, cos, Function, conditional, Mesh, op2)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 5.0
L = 144000.  # Domain length
H = 35000.   # Height position of the model top

if '--running-tests' in sys.argv:
    tmax = dt
    dumpfreq = 1
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
else:
    tmax = 9000.
    dumpfreq = int(tmax / (9*dt))
    nlayers = 70  # horizontal layers
    columns = 180  # number of columns

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
# Make an normal extruded mesh which will be distorted to describe the mountain
m = PeriodicIntervalMesh(columns, L)
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
Vc = VectorFunctionSpace(ext_mesh, "DG", 2)

# Describe the mountain
a = 1000.
xc = L/2.
x, z = SpatialCoordinate(ext_mesh)
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)
zh = 5000.
xexpr = as_vector([x, conditional(z < zh, z + cos(0.5*pi*z/zh)**6*zs, z)])

# Make new mesh
new_coords = Function(Vc).interpolate(xexpr)
mesh = Mesh(new_coords)
mesh._base_mesh = m  # Force new mesh to inherit original base mesh
domain = Domain(mesh, dt, "CG", 1)

# Equation
parameters = CompressibleParameters(g=9.80665, cp=1004.)
sponge = SpongeLayerParameters(H=H, z_level=H-10000, mubar=0.15/dt)
eqns = CompressibleEulerEquations(domain, parameters, sponge=sponge)

# I/O
dirname = 'nonhydrostatic_mountain'
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist=['u'],
                          log_level='INFO',
                          checkpoint_method='dumbcheckpoint')
diagnostic_fields = [CourantNumber(), VelocityZ(), Perturbation('theta'), Perturbation('rho')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
theta_opts = SUPGOptions()
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta", options=theta_opts)]
transport_methods = [DGUpwind(eqns, "u"),
                     DGUpwind(eqns, "rho"),
                     DGUpwind(eqns, "theta", ibp=theta_opts.ibp)]

# Linear solver
linear_solver = CompressibleSolver(eqns)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# spaces
Vu = domain.spaces("HDiv")
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
x, z = SpatialCoordinate(mesh)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(Vt).interpolate(thetab)

# Calculate hydrostatic exner
exner = Function(Vr)
rho_b = Function(Vr)

exner_params = {'ksp_type': 'gmres',
                'ksp_monitor_true_residual': None,
                'pc_type': 'python',
                'mat_type': 'matfree',
                'pc_python_type': 'gusto.VerticalHybridizationPC',
                # Vertical trace system is only coupled vertically in columns
                # block ILU is a direct solver!
                'vert_hybridization': {'ksp_type': 'preonly',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}}

compressible_hydrostatic_balance(eqns, theta_b, rho_b, exner,
                                 top=True, exner_boundary=0.5,
                                 params=exner_params)


def minimum(f):
    fmin = op2.Global(1, [1000], dtype=float)
    op2.par_loop(op2.Kernel("""
static void minify(double *a, double *b) {
    a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
    return fmin.data[0]


p0 = minimum(exner)
compressible_hydrostatic_balance(eqns, theta_b, rho_b, exner,
                                 top=True, params=exner_params)
p1 = minimum(exner)
alpha = 2.*(p1-p0)
beta = p1-alpha
exner_top = (1.-beta)/alpha
compressible_hydrostatic_balance(eqns, theta_b, rho_b, exner,
                                 top=True, exner_boundary=exner_top, solve_for_rho=True,
                                 params=exner_params)

theta0.assign(theta_b)
rho0.assign(rho_b)
u0.project(as_vector([10.0, 0.0]))
remove_initial_w(u0)

stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
