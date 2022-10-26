"""
A moist version of the Wiliamson 5 shallow water test cases (flow over
topography). The moist shallow water framework is that of Bouchut et al.
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, Min)

day = 24*60*60
dt = 900
tmax = 5*dt

# set up shallow water parameters
R = 6371220.
H = 5960.

# set up mesh
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=3)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

# set up moist convective shallow water parameters
q_0 = 3.
alpha = 60.
tau = 200.
gamma = 5.
q_g = 3

parameters = ConvectiveMoistShallowWaterParameters(H=H, gamma=gamma, tau=tau,
                                                   q_0=q_0, alpha=alpha)

dirname = "moist_williamson5"

output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D'],
                          dumpfreq=1,
                          log_level='INFO')

diagnostic_fields = [Sum('D', 'topography')]

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
                                transport_flag=True,
                                transport_eqn=TransportEquationType.advective)

eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr, bexpr=bexpr,
                             active_tracers=[moisture_variable])

# interpolate initial conditions; velocity and height are the same as in
# the dry case and the moisture field is initialised as a constant just below
# saturation
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr
Qexpr = q_g * Constant(1 - 1e-4)

u0 = state.fields('u')
D0 = state.fields('D')
Q0 = state.fields("Q_mixing_ratio")
u0.project(uexpr)
D0.interpolate(Dexpr)
Q0.interpolate(Qexpr)

# Add Bouchut condensation forcing
BouchutForcing(eqns, parameters)

# Build time stepper
stepper = Timestepper(state, ((eqns, RK4(state)),))

stepper.run(t=0, tmax=5*dt)


