"""
The 1 metre high mountain test case. This is solved with the hydrostatic
compressible Euler equations.
"""

from gusto import *
from firedrake import (as_vector, VectorFunctionSpace,
                       PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       exp, pi, cos, Function, conditional, Mesh, sqrt)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 5.0
L = 240000.  # Domain length
H = 50000.   # Height position of the model top

if '--running-tests' in sys.argv:
    tmax = dt
    res = 1
    dumpfreq = 1
else:
    tmax = 15000.
    res = 10
    dumpfreq = int(tmax / (5*dt))


# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
# Make an normal extruded mesh which will be distorted to describe the mountain
nlayers = res*20  # horizontal layers
columns = res*12  # number of columns
m = PeriodicIntervalMesh(columns, L)
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
Vc = VectorFunctionSpace(ext_mesh, "DG", 2)

# Describe the mountain
a = 10000.
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
sponge = SpongeLayerParameters(H=H, z_level=H-20000, mubar=0.3)
eqns = CompressibleEulerEquations(domain, parameters, sponge_options=sponge)

# I/O
dirname = 'hydrostatic_mountain'
output = OutputParameters(
    dirname=dirname,
    dumpfreq=dumpfreq,
    dumplist=['u'],
)
diagnostic_fields = [CourantNumber(), ZComponent('u'), HydrostaticImbalance(eqns),
                     Perturbation('theta'), Perturbation('rho')]
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
params = {'mat_type': 'matfree',
          'ksp_type': 'preonly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.SCPC',
          # Velocity mass operator is singular in the hydrostatic case.
          # So for reconstruction, we eliminate rho into u
          'pc_sc_eliminate_fields': '1, 0',
          'condensed_field': {'ksp_type': 'fgmres',
                              'ksp_rtol': 1.0e-8,
                              'ksp_atol': 1.0e-8,
                              'ksp_max_it': 100,
                              'pc_type': 'gamg',
                              'pc_gamg_sym_graph': True,
                              'mg_levels': {'ksp_type': 'gmres',
                                            'ksp_max_it': 5,
                                            'pc_type': 'bjacobi',
                                            'sub_pc_type': 'ilu'}}}

alpha = 0.51  # off-centering parameter
linear_solver = CompressibleSolver(eqns, alpha, solver_parameters=params,
                                   overwrite_solver_parameters=True)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver,
                                  alpha=alpha)

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
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

# Hydrostatic case: Isothermal with T = 250
x, z = SpatialCoordinate(mesh)
Tsurf = 250.
N = g/sqrt(c_p*Tsurf)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(Vt).interpolate(thetab)

# Calculate hydrostatic exner
exner = Function(Vr)
rho_b = Function(Vr)

exner_surf = 1.0         # maximum value of Exner pressure at surface
max_iterations = 10      # maximum number of hydrostatic balance iterations
tolerance = 1e-7         # tolerance for hydrostatic balance iteration

# Set up kernels to evaluate global minima and maxima of fields
min_kernel = MinKernel()
max_kernel = MaxKernel()

# First solve hydrostatic balance that gives Exner = 1 at bottom boundary
# This gives us a guess for the top boundary condition
bottom_boundary = Constant(exner_surf, domain=mesh)
logger.info(f'Solving hydrostatic with bottom Exner of {exner_surf}')
compressible_hydrostatic_balance(
    eqns, theta_b, rho_b, exner, top=False, exner_boundary=bottom_boundary
)

# Solve hydrostatic balance again, but now use minimum value from first
# solve as the *top* boundary condition for Exner
top_value = min_kernel.apply(exner)
top_boundary = Constant(top_value, domain=mesh)
logger.info(f'Solving hydrostatic with top Exner of {top_value}')
compressible_hydrostatic_balance(
    eqns, theta_b, rho_b, exner, top=True, exner_boundary=top_boundary
)

max_bottom_value = max_kernel.apply(exner)

# Now we iterate, adjusting the top boundary condition, until this gives
# a maximum value of 1.0 at the surface
lower_top_guess = 0.9*top_value
upper_top_guess = 1.2*top_value
for i in range(max_iterations):
    # If max bottom Exner value is equal to desired value, stop iteration
    if abs(max_bottom_value - exner_surf) < tolerance:
        break

    # Make new guess by average of previous guesses
    top_guess = 0.5*(lower_top_guess + upper_top_guess)
    top_boundary.assign(top_guess)

    logger.info(
        f'Solving hydrostatic balance iteration {i}, with top Exner value '
        + f'of {top_guess}'
    )

    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, exner, top=True, exner_boundary=top_boundary
    )

    max_bottom_value = max_kernel.apply(exner)

    # Adjust guesses based on new value
    if max_bottom_value < exner_surf:
        lower_top_guess = top_guess
    else:
        upper_top_guess = top_guess

logger.info(f'Final max bottom Exner value of {max_bottom_value}')

# Perform a final solve to obtain hydrostatically balanced rho
compressible_hydrostatic_balance(
    eqns, theta_b, rho_b, exner, top=True, exner_boundary=top_boundary,
    solve_for_rho=True
)

theta0.assign(theta_b)
rho0.assign(rho_b)
u0.project(as_vector([20.0, 0.0]))
remove_initial_w(u0)

stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
