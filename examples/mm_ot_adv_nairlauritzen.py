from math import pi
from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector, parameters, asin, acos, atan_2, Min, Max, sin, cos
    

parameters["pyop2_options"]["lazy_evaluation"] = False

ref_level = 3
T = 5.0
dt = 0.002*T

R = 1.0

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber()]

dirname = "mm_ot_NLadv_ref%s_dt%s" % (ref_level, dt)
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level,
                             degree=2)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)

timestepping = TimesteppingParameters(dt=dt, move_mesh=True)
output = OutputParameters(dirname=dirname, dumpfreq=1, dumplist_latlon=['D','u'])

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              diagnostic_fields=diagnostic_fields,
              fieldlist=fieldlist)

# interpolate initial conditions
u0 = state.fields.u
D0 = state.fields.D

R0 = Constant(R)

x0, y0, z0 = SpatialCoordinate(mesh)
x = R0*x0/sqrt(x0*x0 + y0*y0 + z0*z0)  # because coords can behave unexpectedly
y = R0*y0/sqrt(x0*x0 + y0*y0 + z0*z0)  # away from nodes, e.g. at quad points
z = R0*z0/sqrt(x0*x0 + y0*y0 + z0*z0)

tc = Constant(0.0)  # Constant to hold the current time
Tc = Constant(T)

k = Constant(2.0)  # strength of deformation
theta_c = Constant(0.0)  # latitude of centres of IC blobs
lamda_c1 = Constant(5.0*pi/6.0)  # longitude of blob 1 centre
lamda_c2 = Constant(7.0*pi/6.0)  # longitude of blob 2 centre
h_max = Constant(1.0)  # height of IC
R_t = Constant(0.5*R)  # base radius of blobs

theta = asin(z/R0)  # latitude
lamda = atan_2(y, x)  # longitude
lamda_prime = lamda - 2*pi*tc/Tc


def dist(lamda_, theta_):
    return acos(sin(theta_)*sin(theta) + cos(theta_)*cos(theta)*cos(lamda - lamda_))


d1 = Min(1.0, pow(dist(lamda_c1, theta_c)/R_t, 2))
d2 = Min(1.0, pow(dist(lamda_c2, theta_c)/R_t, 2))
Dexpr = 0.5*(1.0 + cos(pi*d1)) + 0.5*(1.0 + cos(pi*d2))

u_zonal = R0*(k*pow(sin(lamda_prime), 2)*sin(2*theta)*cos(pi*tc/Tc) + 2*pi*cos(theta)/Tc)
u_merid = R0*k*sin(2*lamda_prime)*cos(theta)*cos(pi*tc/Tc)

cartesian_u_expr = -u_zonal*sin(lamda) - u_merid*sin(theta)*cos(lamda)
cartesian_v_expr = u_zonal*cos(lamda) - u_merid*sin(theta)*sin(lamda)
cartesian_w_expr = u_merid*cos(theta)

uexpr = as_vector((cartesian_u_expr, cartesian_v_expr, cartesian_w_expr))

u0.project(uexpr)
D0.interpolate(Dexpr)

state.initialise({'u': u0, 'D': D0})

# Coriolis expression
Omega = Constant(parameters.Omega)
fexpr = 2*Omega*z/R0
V = FunctionSpace(mesh, "CG", 1)
state.f = Function(V).interpolate(fexpr)  # Coriolis frequency (1/s)

eqn_form = "advective"
Deqn = AdvectionEquation(state, D0.function_space(), equation_form=eqn_form)

state.uexpr = uexpr
state.t_const = tc

advection_dict = {}
advection_dict["D"] = SSPRK3(state, D0, Deqn)
advection_dict["u"] = NoAdvection(state, u0)


def initialise_fn():
    state.fields("u").project(uexpr)
    state.fields("D").interpolate(Dexpr)


monitor = MonitorFunction(state.fields("D"))
mesh_generator = OptimalTransportMeshGenerator(mesh, monitor)

mesh_generator.get_first_mesh(initialise_fn)

# build time stepper
stepper = AdvectionTimestepper(state, advection_dict, mesh_generator=mesh_generator)

stepper.run(t=0, tmax=T)
