from gusto import *
from firedrake import IcosahedralSphereMesh, cos, sin, SpatialCoordinate
import sys

dt = 900.
day = 24.*60.*60.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 14*day

refinements = 4  # number of horizontal cells = 20*(4^refinements)

R = 6371220.
H = 8000.

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

fieldlist = ['u', 'D']
timestepping = TimesteppingParameters(dt=dt, move_mesh=True)
dirname = "mm_ot_sw_W6_ref%s" % refinements
output = OutputParameters(dirname=dirname, dumpfreq=1, dumplist_latlon=['D', 'potential_vorticity'])
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)
pv = PotentialVorticity()
diagnostic_fields = [pv]

state = State(mesh, horizontal_degree=1,
              family="BDM",
              Coriolis=parameters.Omega,
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# interpolate initial conditions
# Initial/current conditions
u0 = state.fields("u")
D0 = state.fields("D")
omega = 7.848e-6  # note lower-case, not the same as Omega
K = 7.848e-6
g = parameters.g
Omega = parameters.Omega

theta, lamda = latlon_coords(mesh)

u_zonal = R*omega*cos(theta) + R*K*(cos(theta)**3)*(4*sin(theta)**2 - cos(theta)**2)*cos(4*lamda)
u_merid = -R*K*4*(cos(theta)**3)*sin(theta)*sin(4*lamda)

uexpr = sphere_to_cartesian(mesh, u_zonal, u_merid)


def Atheta(theta):
    return 0.5*omega*(2*Omega + omega)*cos(theta)**2 + 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 + 26 - 32/(cos(theta)**2))


def Btheta(theta):
    return (2*(Omega + omega)*K/30)*(cos(theta)**4)*(26 - 25*cos(theta)**2)


def Ctheta(theta):
    return 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 - 6)


Dexpr = H + (R**2)*(Atheta(theta) + Btheta(theta)*cos(4*lamda) + Ctheta(theta)*cos(8*lamda))/g

u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 8})
D0.interpolate(Dexpr)

state.initialise([('u', u0),
                  ('D', D0)])

ueqn = EulerPoincare(state, u0.function_space())
Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")

advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

linear_solver = ShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state)


def initialise_fn():
    state.fields("u").project(uexpr, form_compiler_parameters={'quadrature_degree': 8})
    state.fields("D").interpolate(Dexpr)
    pv(state)


pv.setup(state)
monitor = MonitorFunction(pv(state), adapt_to="gradient")
mesh_generator = OptimalTransportMeshGenerator(mesh, monitor)

mesh_generator.get_first_mesh(initialise_fn)

stepper = Timestepper(state, advected_fields, linear_solver,
                      sw_forcing, mesh_generator=mesh_generator)

stepper.run(t=0, tmax=tmax)
