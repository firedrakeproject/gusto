from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    as_vector, VectorFunctionSpace, File, assemble, TestFunction, TrialFunction, action, dx
import itertools
from math import pi
from rk import SSPRK3
from equation import Advection

def get_latlon_coords(mesh):
    coords_orig = mesh.coordinates
    mesh_dg_fs = VectorFunctionSpace(mesh, "DG", 1)
    coords_dg = Function(mesh_dg_fs)
    coords_latlon = Function(mesh_dg_fs)
    par_loop("""
    for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
    dg[i][j] = cg[i][j];
    }
    }
    """, dx, {'dg': (coords_dg, WRITE),
              'cg': (coords_orig, READ)})

    # lat-lon 'x' = atan2(y, x)
    coords_latlon.dat.data[:,0] = np.arctan2(coords_dg.dat.data[:,1], coords_dg.dat.data[:,0])
    # lat-lon 'y' = asin(z/sqrt(x^2 + y^2 + z^2))
    coords_latlon.dat.data[:,1] = np.arcsin(coords_dg.dat.data[:,2]/np.sqrt(coords_dg.dat.data[:,0]**2 + coords_dg.dat.data[:,1]**2 + coords_dg.dat.data[:,2]**2))
    coords_latlon.dat.data[:,2] = 0.0

    kernel = op2.Kernel("""
    #define PI 3.141592653589793
    #define TWO_PI 6.283185307179586
    void splat_coords(double **coords) {
        double diff0 = (coords[0][0] - coords[1][0]);
        double diff1 = (coords[0][0] - coords[2][0]);
        double diff2 = (coords[1][0] - coords[2][0]);

        if (fabs(diff0) > PI || fabs(diff1) > PI || fabs(diff2) > PI) {
            const int sign0 = coords[0][0] < 0 ? -1 : 1;
            const int sign1 = coords[1][0] < 0 ? -1 : 1;
            const int sign2 = coords[2][0] < 0 ? -1 : 1;
            if (sign0 < 0) {
                coords[0][0] += TWO_PI;
            }
            if (sign1 < 0) {
                coords[1][0] += TWO_PI;
            }
            if (sign2 < 0) {
                coords[2][0] += TWO_PI;
            }
        }
    }""", "splat_coords")

    op2.par_loop(kernel, coords_latlon.cell_set,
                 coords_latlon.dat(op2.RW, coords_latlon.cell_node_map()))
    return coords_latlon


R = 6371220.
day = 24.*60.*60.
days = 12.
u_0 = 2*pi*R/(12*day)
dt = 3600.
base_mesh = IcosahedralSphereMesh(radius=R, refinement_level=3)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
base_mesh.init_cell_orientations(global_normal)

fieldlist = ['u','D']
timestepping = TimesteppingParameters(dt=dt)

state = ShallowWaterState(base_mesh, vertical_degree=None, horizontal_degree=1,
                          family="BDM",
                          timestepping=timestepping,
                          fieldlist=fieldlist)

steps = int(days*day/dt + 0.5)

base_mesh.coordinates.dat.data[:] = np.load("meshes/mesh_0.npy")[:]

# interpolate initial conditions
u0, D0 = Function(state.V[0]), Function(state.V[1])
x = SpatialCoordinate(base_mesh)
u_max = Constant(u_0)
R0 = Constant(R)
uexpr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
Dexpr = Expression("R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0)) < rc ? (h0/2.0)*(1 + cos(pi*R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0))/rc)) : 0.0", R=R, rc=R/3., h0=1000., x0=0.0, x1=-R, x2=0.0)
D0.interpolate(Dexpr)
state.initialise([u0, D0])

Vu = VectorFunctionSpace(base_mesh, "CG", 1)
v = Function(Vu)

uadv = Function(state.V[0]).project(uexpr)
ubar = Function(state.V[0]).project(uexpr)

dt = state.timestepping.dt
tmax = days*day
t = 0.

dumpcount = itertools.count()
outfile = File("results/w1/field_output.pvd")
outfile.write(D0, ubar, uadv)
outfilell = File("results/w1/field_output_ll.pvd")
coords_ll = get_latlon_coords(base_mesh)
mesh_ll = Mesh(coords_ll)
D_ll = Function(functionspaceimpl.WithGeometry(D0.function_space(), mesh_ll), val=D0.topological, name='D_ll')
ub_ll = Function(functionspaceimpl.WithGeometry(ubar.function_space(), mesh_ll), val=ubar.topological, name='ub_ll')
u_ll = Function(functionspaceimpl.WithGeometry(uadv.function_space(), mesh_ll), val=uadv.topological, name='u_ll')
outfilell.write(D_ll, ub_ll, u_ll)

mesh0_coords = Function(base_mesh.coordinates.function_space())
mesh0_coords.interpolate(SpatialCoordinate(base_mesh))
mesh0 = Mesh(mesh0_coords)

mesh1_coords = Function(base_mesh.coordinates.function_space())
mesh1_coords.interpolate(SpatialCoordinate(base_mesh))
mesh1 = Mesh(mesh1_coords)

mesh2_coords = Function(base_mesh.coordinates.function_space())
mesh2_coords.interpolate(SpatialCoordinate(base_mesh))
mesh2 = Mesh(mesh2_coords)

deltax = Function(base_mesh.coordinates.function_space())
x = base_mesh.coordinates
x0 = mesh0.coordinates
x1 = mesh1.coordinates
x2 = mesh2.coordinates
equation = Advection(state, ubar, D0.function_space())
timestepper = SSPRK3(D0, equation, dt)

invdt = 1/dt
step = 0
while t < tmax + 0.5*dt:
    t += dt

    deltax.dat.data[:] = np.load("meshes/mesh_" + str(step+1) + ".npy")[:] - np.load("meshes/mesh_" + str(step) + ".npy")[:]
    v.assign(Constant(invdt) * deltax)

    x0.dat.data[:] = x.dat.data[:]
    x2.dat.data[:] = np.load("meshes/mesh_" + str(step+1) + ".npy")[:]
    x1.dat.data[:] = x.dat.data[:] + 0.5*deltax.dat.data[:]
    step += 1
    timestepper.apply(D0, D0, [mesh0, mesh1, mesh2], [x0, x1, x2], t, dt, deltax, v, uexpr, uadv)

    if (next(dumpcount) % 1 == 0):
        outfile.write(D0, ubar, uadv)
        coords_ll = get_latlon_coords(base_mesh)
        mesh_ll.coordinates.dat.data[:] = coords_ll.dat.data[:]
        outfilell.write(D_ll, ub_ll, u_ll)
