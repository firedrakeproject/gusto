from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, Min)

day = 24.*60.*60.
tmax = 0.5*day

ref_level = 3
dt = 400.

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(H=H)

dirname = "sdc_sw_W5_ref%s_dt%s" % (ref_level, dt)
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=3)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

output = OutputParameters(dirname=dirname,
                          steady_state_error_fields=["D"],
                          dumpfreq=1)

diagnostic_fields = [CourantNumber(), Sum('D', 'topography')]

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
eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr, bexpr=bexpr)

# interpolate initial conditions
u0 = state.fields('u')
D0 = state.fields('D')
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

u0.project(uexpr)
D0.interpolate(Dexpr)

M = 3
maxk = 2
scheme = IMEX_SDC(state, M, maxk)
#scheme = IMEX_Euler(state)
timestepper = Timestepper(state, ((eqns, scheme),))
timestepper.run(0, tmax)
