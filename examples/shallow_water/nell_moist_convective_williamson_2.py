"""
A moist convective version of the Williamson 2 shallow water test (steady state
geostrophically-balanced flow). The saturation function depends on height and
on a background theta field (from the thermal version of the test). When xi is
zero vapour is initialised very close to saturation and small overshoots will
generate clouds. When xi is 10e-3 no cloud should be produced.
"""
from gusto import (Domain, ShallowWaterParameters, WaterVapour, CloudWater,
                   Rain, TransportEquationType, ShallowWaterEquations,
                   OutputParameters, CourantNumber, RelativeVorticity,
                   PotentialVorticity, ShallowWaterKineticEnergy,
                   ShallowWaterPotentialEnergy,
                   ShallowWaterPotentialEnstrophy, SteadyStateError, IO,
                   lonlatr_from_xyz, DGUpwind, MoistConvectiveSWSolver,
                   DG1Limiter, ZeroLimiter, MixedFSLimiter,
                   SWSaturationAdjustment, InstantRain, ForwardEuler,
                   SplitPhysicsTimestepper, SSPRK3, TrapeziumRule,
                   SemiImplicitQuasiNewton, xyz_vector_from_lonlatr,
                   Function)
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, sin, cos, exp)

split_physics = False

# ----------------------------------------------------------------- #
# Test case parameters
# ----------------------------------------------------------------- #

# For convergence testing:
#  ref 3, dt 300
#  ref 4, dt 150
#  ref 5, dt 75
#  ref 6, dt 37

day = 24*60*60
dt = 300
tmax = 5*day
ndumps = 5
dumpfreq = int(tmax / (ndumps*dt))
ref = 3
R = 6371220.
u_max = 20.
g = 9.80616
phi_0 = 3e4
H = phi_0/g
# buoyancy parameters for background field
epsilon = 1/300
theta_0 = epsilon*phi_0**2
# moisture parameters
# xi = 10e-3
xi = 0
q0 = 0.007
beta1 = 1600
beta2 = 0
qprecip = 1e-4
gamma_r = 1e-3

# ----------------------------------------------------------------- #
# Set up model objects
# ----------------------------------------------------------------- #

# Domain
mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref, degree=2)
degree = 1
domain = Domain(mesh, dt, "BDM", degree)
x = SpatialCoordinate(mesh)

# Equation
parameters = ShallowWaterParameters(H=H, g=g)
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R

tracers = [WaterVapour(space='DG'), CloudWater(space='DG'),
           Rain(space='DG', transport_eqn=TransportEquationType.no_transport)]

eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                             active_tracers=tracers)

# estimate core count for Pileus
print(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} ')

# IO
dirname = "moist_convective_williamson2_ref3"
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist_latlon=['D', 'D_error'],
                          dump_nc=True,
                          dump_vtus=True)

diagnostic_fields = [CourantNumber(), RelativeVorticity(),
                     PotentialVorticity(),
                     ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(parameters),
                     ShallowWaterPotentialEnstrophy(),
                     SteadyStateError('u'), SteadyStateError('D'),
                     SteadyStateError('water_vapour'),
                     SteadyStateError('cloud_water')]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# for the saturation function and time-varying gamma_v
lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])
w = Omega*R*u_max + (u_max**2)/2
sigma = w/10
numerator = theta_0 + sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))
denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2
theta = numerator/denominator
tpexpr = 0  # no topography 


def sat_func(x_in):
    D = x_in.split()[1]
    return (q0*H/(D + tpexpr)) * exp(20*(theta))


# Feedback proportionality is dependent on D
def gamma_v(x_in):
    qsat = sat_func(x_in)
    D = x_in.split()[1]
    return (1 + qsat*(20*beta2/(g) + beta1/(D + tpexpr)))**(-1)


transport_methods = [DGUpwind(eqns, 'u'),
                     DGUpwind(eqns, 'D'),
                     DGUpwind(eqns, 'water_vapour'),
                     DGUpwind(eqns, 'cloud_water')]

linear_solver = MoistConvectiveSWSolver(eqns)

# limiters
DG1limiter = DG1Limiter(domain.spaces('DG'))
zerolimiter = ZeroLimiter(domain.spaces('DG'))
physics_sublimiters = {'water_vapour': zerolimiter,
                       'cloud_water': zerolimiter,
                       'rain': zerolimiter}
transport_sublimiters = {'water_vapour': DG1limiter,
                         'cloud_water': DG1limiter}
physics_limiter = MixedFSLimiter(eqns, physics_sublimiters)
transport_limiter = MixedFSLimiter(eqns, transport_sublimiters)

# physics
sat_adj = SWSaturationAdjustment(eqns, sat_func,
                                 time_varying_saturation=True,
                                 convective_feedback=True, beta1=beta1,
                                 gamma_v=gamma_v, time_varying_gamma_v=True,
                                 parameters=parameters)
inst_rain = InstantRain(eqns, qprecip, vapour_name="cloud_water",
                        rain_name="rain", gamma_r=gamma_r)

physics_schemes = [(sat_adj, ForwardEuler(domain, limiter=physics_limiter)),
                   (inst_rain, ForwardEuler(domain, limiter=physics_limiter))]

# timestepper
if split_physics:
    stepper = SplitPhysicsTimestepper(eqns, SSPRK3(domain,
                                                   limiter=transport_limiter),
                                      io,
                                      spatial_methods=transport_methods,
                                      physics_schemes=physics_schemes)

else:
    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "D"),
                          SSPRK3(domain, "water_vapour", limiter=DG1limiter),
                          SSPRK3(domain, "cloud_water", limiter=DG1limiter),
                          ]
    stepper = SemiImplicitQuasiNewton(eqns, io,
                                      transport_schemes=transported_fields,
                                      spatial_methods=transport_methods,
                                      linear_solver=linear_solver,
                                      physics_schemes=physics_schemes,
                                      num_outer=2, num_inner=2)

# ----------------------------------------------------------------- #
# Initial conditions
# ----------------------------------------------------------------- #

u0 = stepper.fields("u")
D0 = stepper.fields("D")
v0 = stepper.fields("water_vapour")

uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, x)

Dexpr = H - (1/g)*(w)*((sin(phi))**2)

# though this set-up has no buoyancy, we use the expression for theta to set up
# the initial vapour
initial_msat = q0*g*H/(g*Dexpr) * exp(20*theta)
vexpr = (1 - xi) * initial_msat

u0.project(uexpr)
D0.interpolate(Dexpr)
v0.interpolate(vexpr)

# Set reference profiles
Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)