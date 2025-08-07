"""
An implementation of the moist thermal version of the moist gravity wave test
for comparison with the moist dynamics formulation version of the same test. The
set-up is based on the set-up of the moist dynamics formulation for a fair
comparison.
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, pi, sqrt,
                       min_value, cos, Constant, exp)

# ----------------------------------------------------------------- #
# Test case parameters
# ----------------------------------------------------------------- #

# For convergence tests use dt = 50, 100, 200, 400, 600

physics_beta = 1.0
day = 24*60*60
dt = 3600
tmax = 5*day
ref = 3
R = 6371220.
H = 5960.
u_max = 20.
ndumps = 20
dumpfreq = int(tmax / (ndumps*dt))
# moist shallow water parameters
epsilon = 1/300
q0 = 0.0115
beta2 = 9.80616*10
gamma_v = 1
nu = 1.5
# perturbation parameters
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
phi_c = pi/6.
# parameters for initial buoyancy
phi_0 = 3e4
epsilon = 1/300
theta_0 = epsilon*phi_0**2

# Domain
mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref, degree=2)
degree = 1
domain = Domain(mesh, dt, "BDM", degree)
x = SpatialCoordinate(mesh)

# Equation parameters
parameters = ShallowWaterParameters(mesh, H=H, nu=nu, beta2=beta2, q0=q0)
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R

# Perturbation
lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])
lsq = (lamda - lamda_c)**2
thsq = (phi - phi_c)**2
rsq = min_value(R0sq, lsq+thsq)
r = sqrt(rsq)
pert = 2000.0 * (1 - r/R0)

# Equation
tracers = [WaterVapour(space='DG'), CloudWater(space='DG')]
eqns = ThermalShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                    active_tracers=tracers)

# estimate core count for Pileus
print(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} ')

# IO
dirname = "moist_thermal_gravity_wave_inner_phys_4x1"
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist_latlon=['D'],
                          dump_nc=True,
                          dump_vtus=True,
                          checkpoint=True,
                          chkptfreq=dumpfreq)

diagnostic_fields = [CourantNumber(),
                     RelativeVorticity(), PotentialVorticity()]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

transport_methods = [DGUpwind(eqns, field_name) for field_name in eqns.field_names]

linear_solver = MoistThermalSWSolver(eqns, physics_beta=physics_beta)

# limiters
DG1limiter = DG1Limiter(domain.spaces('DG'))
zerolimiter = ZeroLimiter(domain.spaces('DG'))

physics_sublimiters = {'water_vapour': zerolimiter,
                       'cloud_water': zerolimiter}

transport_sublimiters = {'b': DG1limiter,
                         'water_vapour': DG1limiter,
                         'cloud_water': DG1limiter}

transport_limiter = MixedFSLimiter(eqns, transport_sublimiters)
physics_limiter = MixedFSLimiter(eqns, physics_sublimiters)

# physics
# saturation function (depending on b_e)
def sat_func(D, b_e):
    return (q0*H/D) * exp(nu*(1 - b_e/g))

# saturation function to pass to physics (takes mixed function as argument)
def phys_sat_func(x_in):
    D = x_in.subfunctions[1]
    b = x_in.subfunctions[2]
    q_v = x_in.subfunctions[3]
    b_e = Function(b.function_space()).interpolate(b - beta2*q_v)
    return (q0*H/D) * exp(nu*(1 - b_e/g))


# Physics schemes
sat_adj = SWSaturationAdjustment(eqns, phys_sat_func,
                                 time_varying_saturation=True,
                                 parameters=parameters,
                                 thermal_feedback=True,
                                 beta2=beta2, gamma_v=gamma_v)

physics_schemes = [(sat_adj, ForwardEuler(domain,
                                          limiter=physics_limiter))]

# ----------------------------------------------------------------- #
# Timestepper
# ----------------------------------------------------------------- #

transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D"),
                      SSPRK3(domain, "b", limiter=DG1limiter),
                      SSPRK3(domain, "water_vapour", limiter=DG1limiter),
                      SSPRK3(domain, "cloud_water", limiter=DG1limiter)]
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver,
                                  physics_schemes=physics_schemes,
                                  num_outer=4, num_inner=1,
                                  physics_beta=physics_beta)

# ----------------------------------------------------------------- #
# Initial conditions
# ----------------------------------------------------------------- #

u0 = stepper.fields("u")
D0 = stepper.fields("D")
b0 = stepper.fields("b")
v0 = stepper.fields("water_vapour")
c0 = stepper.fields("cloud_water")

# velocity
uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, x)

# buoyancy
g = parameters.g
w = Omega*R*u_max + (u_max**2)/2
sigma = w/10
numerator = theta_0 + sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))
denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2
theta = numerator/denominator
b_guess = parameters.g * (1 - theta)

# depth
Dexpr = H - (1/g)*(w + sigma)*((sin(phi))**2) + pert

# iterate to find initial b_e_expr, from which the vapour and saturation
# function are recovered
q_t = 0.03
def iterate():
    n_iterations = 10
    D_init = Function(D0.function_space()).interpolate(Dexpr)
    b_init = Function(b0.function_space()).interpolate(b_guess)
    b_e_init = Function(b0.function_space()).interpolate(b_init - beta2*q_t)
    q_v_init = Function(v0.function_space()).interpolate(q_t)
    for i in range(n_iterations):
        q_sat_expr = sat_func(D_init, b_e_init)
        dq_sat_dq_v_expr = nu*beta2/g*q_sat_expr
        q_v_init.interpolate(q_v_init - (q_sat_expr - q_v_init)/(dq_sat_dq_v_expr - 1.0))
        b_e_init.interpolate(b_init - beta2*q_v_init)
    return b_e_init

b_e = iterate()

initial_sat = sat_func(Dexpr, b_e)

vexpr = initial_sat

# back out the initial buoyancy using b_e and q_v
bexpr = b_e + beta2*vexpr

# cloud is the rest of q_t that isn't vapour
cexpr = Constant(q_t) - vexpr

u0.project(uexpr)
D0.interpolate(Dexpr)
b0.interpolate(bexpr)
v0.interpolate(vexpr)
c0.interpolate(cexpr)

# Set reference profiles
Dbar = Function(D0.function_space()).assign(H)
bbar = Function(b0.function_space()).interpolate(bexpr)
vbar = Function(v0.function_space()).interpolate(vexpr)
cbar = Function(c0.function_space()).interpolate(cexpr)
stepper.set_reference_profiles([('D', Dbar), ('b', bbar),
                                ('water_vapour', vbar), ('cloud_water', cbar)])

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)