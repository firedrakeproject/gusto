from math import pi
import sys
from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, parameters, acos, Min, \
    sin, cos, SpatialCoordinate


parameters["pyop2_options"]["lazy_evaluation"] = False

ref_level = 3
T = 5.0
dt = 0.002*T
if '--running-tests' in sys.argv:
    T = dt

R = 1.0

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters()
diagnostic_fields = [CourantNumber()]

dirname = "mm_ot_NLadv_ref%s_dt%s" % (ref_level, dt)
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level,
                             degree=2)
mesh.init_cell_orientations(SpatialCoordinate(mesh))

timestepping = TimesteppingParameters(dt=dt, move_mesh=True)
output = OutputParameters(dirname=dirname, dumpfreq=1, dumplist_latlon=['D', 'u'])

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostic_fields=diagnostic_fields,
              fieldlist=fieldlist)

# interpolate initial conditions
u0 = state.fields("u")
D0 = state.fields("D")

R0 = Constant(R)

tc = Constant(0.0)  # Constant to hold the current time
Tc = Constant(T)

k = Constant(2.0)  # strength of deformation
theta_c = Constant(0.0)  # latitude of centres of IC blobs
lamda_c1 = Constant(5.0*pi/6.0)  # longitude of blob 1 centre
lamda_c2 = Constant(7.0*pi/6.0)  # longitude of blob 2 centre
h_max = Constant(1.0)  # height of IC
R_t = Constant(0.5*R)  # base radius of blobs

theta, lamda = latlon_coords(mesh)
lamda_prime = lamda - 2*pi*tc/Tc


def dist(lamda_, theta_):
    return R0*acos(sin(theta_)*sin(theta) + cos(theta_)*cos(theta)*cos(lamda - lamda_))


d1 = Min(1.0, dist(lamda_c1, theta_c)/R_t)
d2 = Min(1.0, dist(lamda_c2, theta_c)/R_t)
Dexpr = 0.5*(1.0 + cos(pi*d1)) + 0.5*(1.0 + cos(pi*d2))

u_zonal = R0*(k*pow(sin(lamda_prime), 2)*sin(2*theta)*cos(pi*tc/Tc) + 2*pi*cos(theta)/Tc)
u_merid = R0*k*sin(2*lamda_prime)*cos(theta)*cos(pi*tc/Tc)

uexpr = sphere_to_cartesian(mesh, u_zonal, u_merid)

u0.project(uexpr)
D0.interpolate(Dexpr)

state.initialise([('u', u0), ('D', D0)])

eqn_form = "advective"
Deqn = AdvectionEquation(state, D0.function_space(), equation_form=eqn_form)

state.uexpr = uexpr
state.t_const = tc

advected_fields = []
advected_fields.append(("D", SSPRK3(state, D0, Deqn)))
advected_fields.append(("u", NoAdvection(state, u0)))


def initialise_fn():
    state.fields("u").project(uexpr)
    state.fields("D").interpolate(Dexpr)


monitor = MonitorFunction(state.fields("D"), adapt_to="hessian", avg_weight=0.5, max_min_cap=4.0)
mesh_generator = OptimalTransportMeshGenerator(mesh, monitor)

mesh_generator.get_first_mesh(initialise_fn)

# build time stepper
stepper = AdvectionDiffusion(state, advected_fields, mesh_generator=mesh_generator)

stepper.run(t=0, tmax=T)
