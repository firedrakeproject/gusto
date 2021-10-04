from math import pi
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       cos, sin, acos, conditional, Constant, exp, Function)

nondivergent = False
SSHS_ms = False
SSHS_lat_only = False

# parameters
R = 6371220.
day = 24.*60.*60.

# set up mesh
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=3)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

# lat lon co-ordinates
theta, lamda = latlon_coords(mesh)

timestepping = TimesteppingParameters(dt=3000.)
dirname = 'deformational_moisture'
fieldlist = ['u', 'D']

output = OutputParameters(dirname=dirname,
                          dumpfreq=23)

diagnostics = Diagnostics("u", "m1", "m2")

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              fieldlist=fieldlist,
              diagnostics=diagnostics)

# set up initial conditions
u0 = state.fields("u")
D0 = state.fields("D")

# parameters for initial conditions from Lauritzen
u_max = 2*pi*R/(12*day)
alpha = 0.
# initial height depends on the choice of saturation field
if SSHS_ms:
    if SSHS_lat_only:
        scale = 1
    else:
        scale = 0.087
else:
    scale = 0.032
h_max = 1
b = 0.1
c = 0.9
lamda_1 = 2*pi/2 - pi/6
lamda_2 = 2*pi/2 + pi/6
lamda_c = 3*pi/2
theta_c = 0
br = R/2
r1 = R * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_1))
r2 = R * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_2))
h1expr = b + c * (h_max/2)*(1 + cos(pi*r1/br))
h2expr = b + c * (h_max/2)*(1 + cos(pi*r2/br))

# velocity parameters
T = 12*day

if nondivergent:
    def uexpr(t):
        lamda_p = lamda - 2*pi*t/T
        u_zonal = (
            10*R/T * (sin(lamda_p))**2 * sin(2*theta) * cos(pi*t/T)
            + 2*pi*R/T * cos(theta)
        )
        u_merid = 10*R/T * sin(2*lamda_p) * cos(theta) * cos(pi*t/T)
        return sphere_to_cartesian(mesh, u_zonal, u_merid)
else:
    def uexpr(t):
        lamda_p = lamda - 2*pi*t/T
        u_zonal = (
            -5*R/T * (sin(lamda_p/2))**2 * sin(2*theta) *
            (cos(theta))**2 * cos(pi*t/T) + 2*pi*R/T * cos(theta)
        )
        u_merid = 5/2*R/T * sin(lamda_p) * (cos(theta))**3 * cos(pi*t/T)
        return sphere_to_cartesian(mesh, u_zonal, u_merid)

u0.project(uexpr(0))

# set up advected variable in the same space as the height field
VD = D0.function_space()
m1 = state.fields("m1", space=VD)
m2 = state.fields("m2", space=VD)

mtot = state.fields("mtot", space=VD)

# choose saturation profile: SSHS 2021 paper OR varying in latitude
if SSHS_ms:
    if SSHS_lat_only:
        alpha_0 = 0
    else:
        alpha_0 = 1
    ms = exp(-(theta**2/(pi/3)**2) - alpha_0 * (lamda - pi)**2/(2*pi/3)**2)

else: # varying in latitude
    Gamma = Constant(-37.5604) # lapse rate - this could be lat-dependent
    T0 = Constant(300)   # temperature at equator
    T = Gamma * abs(theta) + T0   # temperature profile
    e1 = Constant(0.98)  # level of saturation
    ms = 3.8e-3 * exp((18 * T - 4824)/(T - 30))   # saturation profile
    # set up a temperature field for viewing the temperature profile
    temp_field = state.fields("temp", space=VD)
    temp_field.interpolate(T)

# initialise m1 as the two cosine bells
m1.interpolate(scale * (conditional(r1 < br, h1expr, conditional(r2 < br, h2expr, b))))

m1eqn = AdvectionEquation(state, VD, equation_form="advective")
mtot.interpolate(m1 + m2)

m1eqn = AdvectionEquation(state, VD, equation_form="advective")
m2eqn = AdvectionEquation(state, VD, equation_form="advective")

advected_fields = []
advected_fields.append(("m1", SSPRK3(state, m1, m1eqn)))
advected_fields.append(("m2", SSPRK3(state, m2, m1eqn)))


class Moisture(Physics):
    def __init__(self, state, ms):
        super().__init__(state)
        V = state.fields("m1").function_space()
        self.dm1 = Function(V)
        self.dm2 = Function(V)
        self.ms = ms

    def apply(self):
        ms = self.ms
        m1 = state.fields("m1")
        m2 = state.fields("m2")
        gamma1 = Constant(0.9)
        dt = state.timestepping.dt
        self.dm1.interpolate(conditional(m1 - ms > 0, gamma1 * (m1 - ms), 0))
        self.dm2.interpolate(
            conditional(ms - m1 > 0,
                        conditional(ms - m1 < m2, gamma1 * (ms - m1), m2),
                        0))
        m1 += self.dm2 - self.dm1
        m2 += self.dm1 - self.dm2
        mtot.interpolate(m1 + m2)

moisture = Moisture(state, ms)

# build time stepper
timestepper = AdvectionDiffusion(state, advected_fields,
                                 physics_list=[moisture])
timestepper.run(t=0, tmax=12*day)
