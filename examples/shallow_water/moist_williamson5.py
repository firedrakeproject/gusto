"""
A moist version of the Williamson 5 shallow water test cases (flow over
topography). The moist shallow water framework is that of Bouchut et al.
"""
split_physics = True

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, Min, exp, conditional, cos, acos)

day = 24*60*60
dt = 300
tmax = 50*day

# set up shallow water parameters
R = 6371220.
H = 5960.

# set up mesh
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=1)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

# set up moist convective shallow water parameters
q_0 = 3.
alpha = -0.6
tau = 200.
gamma = 5.
q_g = 3

parameters = ConvectiveMoistShallowWaterParameters(H=H, gamma=gamma, tau=tau,
                                                   q_0=q_0, alpha=alpha)

dirname = "moist_williamson5_temp_split"

ndumps = 50
dumpfreq = int(tmax / (ndumps*dt))

output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D'],
                          dumpfreq=1,
                          log_level='INFO')

diagnostic_fields = [Sum('D', 'topography'), CourantNumber()]

state = State(mesh,
              dt=dt,
              output=output,
              parameters=parameters,
              diagnostic_fields=diagnostic_fields)

Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
theta, lamda = latlon_coords(mesh)
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
lsq = (lamda - lamda_c)**2
theta_c = pi/6.
thsq = (theta - theta_c)**2
rsq = Min(R0sq, lsq+thsq)
r = sqrt(rsq)
bexpr = 2000 * (1 - r/R0)

moisture_variable = WaterVapour(name="Q", space="DG",
                                variable_type=TracerVariableType.mixing_ratio,
                                transport_eqn=TransportEquationType.conservative)

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr, bexpr=bexpr,
                             active_tracers=[moisture_variable])

# interpolate initial conditions; velocity and height are the same as in
# the dry case
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr
# initial moisture is cosine blob from deformational test
q_max = 1
b = 0.1
c = 0.9
lamda_1 = 7*pi/6
theta_c = pi/6
br = R/4
r1 = R * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_1))
q1expr = b + c * (q_max/2)*(1 + cos(pi*r1/br))

u0 = state.fields('u')
D0 = state.fields('D')
Q0 = state.fields("Q_mixing_ratio")
u0.project(uexpr)
D0.interpolate(Dexpr)
Q0.interpolate(conditional(r1 < br, 3*q1expr, b))

# define saturation function
saturation = q_0 * exp(-alpha*(state.fields("D")-H)/H)

#  define timestepper based on whether physics is being stepped separately to
#  the dynamics
if split_physics:
    physics_schemes = [(InstantRain(eqns, saturation,
                                    vapour_name="Q_mixing_ratio",
                                    parameters=parameters,
                                    convective_feedback=True),
                        ForwardEuler(state))]
    stepper = SplitPhysicsTimestepper(eqns, RK4(state), state,
                                      physics_schemes=physics_schemes)

else:
    InstantRain(eqns, saturation, vapour_name="Q_mixing_ratio",
                parameters=parameters, convective_feedback=True)
    stepper = Timestepper(eqns, RK4(state), state)

stepper.run(t=0, tmax=5*dt)
