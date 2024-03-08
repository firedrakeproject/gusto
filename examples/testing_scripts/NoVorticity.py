
from gusto import (Domain, CompressibleParameters, OutputParameters, ZonalComponent, 
                   MeridionalComponent, RadialComponent, CompressibleSolver, SSPRK3, 
                   DGUpwind,  CompressibleEulerEquations, 
                   IO, Timestepper, RK4, GeneralCubedSphereMesh, lonlatr_from_xyz, 
                   xyz_vector_from_lonlatr)
from firedrake import (PeriodicRectangleMesh, ExtrudedMesh, Constant,
                       SpatialCoordinate, Function, as_vector )
from icecream import ic

from gusto.diagnostics import CompressibleRelativeVorticity, CompressibleAbsoluteVorticity

days = 5 # suggested is 15
d = 24*3600*3600
dt = 0.001
tmax = dt * 10
Lx = 1 # length
Ly = 1 # width
H = 5  # height
degree = 1
dumpfreq = 1
nlayers = 5
ncolumnsx = 5
ncolumnsy = 5
n = 8
m = GeneralCubedSphereMesh(2, num_cells_per_edge_of_panel=n, degree=2)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers, extrusion_type='radial')
domain = Domain(mesh, dt, "RTCF", degree=degree)



# CompressibleParameters does not have an omega as in 2Ω×u where Ω = (0,0,omega),
# therefore we manualy define it before and pass the vector form of Omega to the eqns class
# my concern is that whilst i can hard code this value (earths) if a user wanted to change it 
# im unsure how it would pass to the diagnostics
omega = Constant(7.292e-5)
params = CompressibleParameters(Omega=omega)
eqns = CompressibleEulerEquations(domain, params)
ic(f'ideal number of processors = {eqns.X.function_space().dim() / 50000}')

dirname = f'omega_passing'
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq, dump_nc = True)
diagnostic_fields = [CompressibleRelativeVorticity(), MeridionalComponent('CompressibleRelativeVorticity'), 
                     ZonalComponent('CompressibleRelativeVorticity'), RadialComponent('CompressibleRelativeVorticity'),
                     CompressibleAbsoluteVorticity(params), MeridionalComponent('CompressibleAbsoluteVorticity'), 
                    ZonalComponent('CompressibleAbsoluteVorticity'), RadialComponent('CompressibleAbsoluteVorticity'),
                     ZonalComponent('u'), MeridionalComponent('u'), RadialComponent('u')]
io=IO(domain, output, diagnostic_fields=diagnostic_fields)

transported_fields = [SSPRK3(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta")]

transport_methods =  [DGUpwind(eqns, 'u'),
                      DGUpwind(eqns, 'rho'),
                      DGUpwind(eqns, 'theta')]

linear_solver = CompressibleSolver(eqns)
stepper = Timestepper(eqns, RK4(domain), io)


xyz = SpatialCoordinate(mesh)
lon, lat_, r = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
e_lon = xyz_vector_from_lonlatr(1, 0, 0, xyz)
e_lat = xyz_vector_from_lonlatr(0, 1, 0, xyz)
e_r = xyz_vector_from_lonlatr(0, 0, 1, xyz)

zonal_u = 0
meridional_u = 0
radial_u = H - r

u = stepper.fields('u')
rho = stepper.fields("rho")
theta = stepper.fields("theta")
Vu = u.function_space()
Vt = theta.function_space()
Vr = rho.function_space()


ic('projecting u and setting reference profiles')
uexpr = zonal_u * e_lon + meridional_u *e_lat + radial_u * e_r
u.project(uexpr)
rho_b = Function(Vr).assign(rho)
theta_b = Function(Vt).assign(theta)
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])
ic('Initialising stepper')
stepper.run(t=0, tmax=tmax)