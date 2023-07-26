from gusto import *
from firedrake import (IcosahedralSphereMesh, acos, sin, cos, Constant, norm)

process = "condensation"

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Parameters
dt = 100
R = 6371220.
H = 100
theta_c = pi
lamda_c = pi/2
rc = R/4
L = 10
beta2 = 1

# Domain
mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=2)
degree = 1
domain = Domain(mesh, dt, 'BDM', degree)
x = SpatialCoordinate(mesh)
theta, lamda = latlon_coords(mesh)

# saturation field (constant everywhere)
sat = 100

# Equation
parameters = ShallowWaterParameters(H=H)
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R

tracers = [WaterVapour(space='DG'), CloudWater(space='DG')]

eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                             u_transport_option='vector_advection_form',
                             thermal=True, active_tracers=tracers)

# I/O
output = OutputParameters(dirname="cond_evap_testing",
                          dumpfreq=1)
io = IO(domain, output, diagnostic_fields=[Sum('water_vapour', 'cloud_water')])

# Physics schemes
physics_schemes = [(SW_SaturationAdjustment(eqns, sat, L=L,
                                            parameters=parameters,
                                            thermal_feedback=True,
                                            beta2=beta2),
                    ForwardEuler(domain))]

# Time stepper
scheme = ForwardEuler(domain)

stepper = SplitPhysicsTimestepper(eqns, scheme, io,
                                  physics_schemes=physics_schemes)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields("u")
D0 = stepper.fields("D")
b0 = stepper.fields("b")
v0 = stepper.fields("water_vapour")
c0 = stepper.fields("cloud_water")

# perturbation
r = R * (
    acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_c)))
pert = conditional(r < rc, 1.0, 0.0)

if process == "evaporation":
    # atmosphere is subsaturated and cloud is present
    v0.interpolate(0.96*Constant(sat))
    c0.interpolate(0.005*sat*pert)
    # lose cloud and add this to vapour
    v_true = Function(v0.function_space()).interpolate(sat*(0.96+0.005*pert))
    c_true = Function(c0.function_space()).interpolate(Constant(0.0))
    # lose buoyancy (sat_adj_expr is -0.005 here)
    factor = Constant(parameters.g*L*beta2)
    sat_adj_expr = -0.005
    b_true = Function(b0.function_space()).interpolate(factor*sat_adj_expr)

elif process == "condensation":
    # vapour is above saturation
    v0.interpolate(sat*(1.0 + 0.04*pert))
    # lose vapour and add this to cloud
    v_true = Function(v0.function_space()).interpolate(Constant(sat))
    c_true = Function(c0.function_space()).interpolate(v0 - sat)
    # gain buoyancy (sat_adj_expr is 0.04 here)
    factor = Constant(parameters.g*L*beta2)
    sat_adj_expr = 0.004
    b_true = Function(b0.function_space()).interpolate(factor*sat_adj_expr)

c_init = Function(c0.function_space()).interpolate(c0)
print("initial vapour:")
print(v0.dat.data.max())
print("initial cloud:")
print(c0.dat.data.max())

stepper.run(t=0, tmax=dt)

vapour = stepper.fields("water_vapour")
cloud = stepper.fields("cloud_water")
buoyancy = stepper.fields("b")

assert norm(vapour - v_true) / norm(v_true) < 0.001, \
    f'Final vapour field is incorrect for {process}'

# Check that cloud has been created / removed
denom = norm(c_true) if process == "condensation" else norm(c_init)
assert norm(cloud - c_true) / denom < 0.001, \
    f'Final cloud field is incorrect for {process}'

###################
# printing to try
###################

print(process)

# if norm(vapour - v_true) / norm(v_true) < 0.001:
    # print("passed vapour!")
    # print(norm(vapour - v_true) / norm(v_true))

denom = norm (c_true) if process == "condensation" else norm(c_init)
# if norm(cloud - c_true) / denom < 0.001:
    # print("passed cloud!")
    # print(norm(cloud - c_true) / denom)

print("true buoyancy:")
print(b_true.dat.data.max())
print("b field:")
print(buoyancy.dat.data.max())
# print("norm:")
# print(norm(buoyancy - b_true))

print("vapour after:")
print(vapour.dat.data.max())
print("cloud after:")
print(cloud.dat.data.max())
