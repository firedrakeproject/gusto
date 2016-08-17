from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector
from math import pi
import os
# setup resolution and timestepping parameters for convergence test
# ref_dt = {3:3000., 4:1500., 5:750., 6:375}
ref_dt = {3:7200.}

# setup shallow water parameters
R = 6371220.
day = 24.*60.*60.
days = 12.
u_0 = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters()
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.iteritems():

    dirname = "sw_mmalt_W1_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level)

    steps = int(days*day/dt + 0.5)

    speed = 10.0
    theta = speed*dt/R
    c, s = np.cos(theta), np.sin(theta)
    mat = np.matrix([[c, -s], [s, c]])
    directory = "results/" + dirname + "meshes/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for ii in range(steps + 1):
        np.save("results/" + dirname + "meshes/mesh_" + str(ii) + ".npy", mesh.coordinates.dat.data)
        mesh.coordinates.dat.data[:, 1:3] = np.dot(mesh.coordinates.dat.data[:, 1:3], mat)

    mesh.coordinates.dat.data[:] = np.load("results/" + dirname + "meshes/mesh_0.npy")[:]

    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname, dumpfreq=12, dumplist_latlon=['D','u'], Verbose=True)

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters,
                              diagnostics=diagnostics,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0, D0 = Function(state.V[0]), Function(state.V[1])
    x = SpatialCoordinate(mesh)
    u_max = Constant(u_0)
    R0 = Constant(R)
    uexpr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
    Dexpr = Expression("R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0)) < rc ? (h0/2.0)*(1 + cos(pi*R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0))/rc)) : 0.0", R=R, rc=R/3., h0=1000., x0=0.0, x1=-R, x2=0.0)
    D0.interpolate(Dexpr)
    state.initialise([u0, D0])

    # Coriolis expression
    Omega = Constant(parameters.Omega)
    fexpr = 2*Omega*x[2]/R0
    V = FunctionSpace(mesh, "CG", 1)
    state.f = Function(V).interpolate(fexpr)  # Coriolis frequency (1/s)

    advection_dict = {}
    advection_dict["D"] = DGAdvection(state, state.V[1], continuity=False)

    # build time stepper
    Vu = VectorFunctionSpace(mesh, "CG", 1)
    uadv = Function(Vu)

    step = 0
    def meshx_callback(self):
        global step
        step += 1
        self.oldx.assign(self.state.mesh.coordinates)
        self.state.mesh.coordinates.dat.data[:] = np.load("results/" + dirname + "meshes/mesh_" + str(step) + ".npy")[:]

    invdt = 1/dt
    firsthalf = True
    def meshv_callback(self):
        global firsthalf
        if firsthalf:
            self.deltax.dat.data[:] = np.load("results/" + dirname + "meshes/mesh_" + str(step+1) + ".npy")[:] - np.load("results/" + dirname + "meshes/mesh_" + str(step) + ".npy")[:]
        else:
            pass

        self.mesh_velocity.assign(Constant(invdt) * self.deltax)
        firsthalf = not firsthalf
            

    moving_mesh_advection = MovingMeshAdvection(state, advection_dict, meshx_callback, meshv_callback, uadv=uadv, uexpr=uexpr)
    stepper = MovingMeshAdvectionTimestepper(state, advection_dict, moving_mesh_advection)

    stepper.run(t=0, tmax=days*day)
