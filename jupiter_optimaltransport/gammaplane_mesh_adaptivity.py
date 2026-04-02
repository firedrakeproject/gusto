# make a mesh with higher resolution over where the vortex is

from firedrake import *

# set up the initial square mesh
Lx = 7e7
Ly = Lx 
nx = 256
ny = nx 

mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
x = SpatialCoordinate(mesh)

# moves the pole to the centre of the mesh 
mesh.coordinates.dat.data[:, 0] -= 0.5 * Lx
mesh.coordinates.dat.data[:, 1] -= 0.5 * Lx 


# set up the two densities:
mu_expression = conditional((sqrt(x[0]**2 + x[1]**2) < 1e7), Constant(1), Constant(0.0000000001))
nu_expression = conditional((sqrt(x[0]**2 + x[1]**2) < 2.5e7), Constant(1), Constant(0.0000000001))


# set up the parameters for the sinkhorn iterations
epsilon = 0.02 * Lx * 2e7   
steps = 30
delta_t = epsilon/steps

V = FunctionSpace(mesh, "CG", 1)

phi = Function(V, name="phi")
psi = Function(V, name="psi")

v = TestFunction(V)
u = TrialFunction(V) 

# creates the functions mu and nu 
mu = Function(V, name="mu")
nu = Function(V, name="nu")
mu.interpolate(mu_expression) 
nu.interpolate(nu_expression)

# ensures their integrals have the same value 
integral_mu = assemble(mu * dx)  # calculates the integral of mu dx
integral_nu = assemble(nu * dx)

area_value = Lx * Lx 
mu.assign(mu * area_value/integral_mu) 
nu.assign(nu * area_value/integral_nu)

# initialise phi and psi to 1 
phi.interpolate(1)  
psi.interpolate(1) 

print(f"epsilon = {epsilon}, delta_t = {delta_t}")

# for observing convergence
err_phi = 1.0
prev_phi = Function(V)
prev_phi.assign(phi)

err_psi = 1.0
prev_psi = Function(V)
prev_psi.assign(psi)

n = 0

# set up the PDE problem to solve the diffusion eqn 
a = (Constant(delta_t) * inner(grad(u), grad(v)) + inner(u, v)) * dx 
L1 = inner(phi, v) * dx 
L2 = inner(psi, v) * dx

u = Function(V)

# run a set number of sinkhorn iterations, or can be changed
# to run to convergence of a given tolerance depending on err_phi
for _ in range(100):

    for i in range(steps):

        solve(a == L1, u)
        phi.assign(u)  # makes phi = next value of phi

    # psi_{n+1} = nu/phi_{n}
    psi.interpolate(nu/phi)

    for i in range(steps):

        solve(a == L2, u)
    
        psi.assign(u)  # makes psi = next value of psi

    # phi_{n+1} = mu/psi_{n+1}
    phi.interpolate(mu/psi)


    # for testing convergence 
    err_phi = norm(phi - prev_phi) 
    err_psi = norm(psi - prev_psi)
    print(f"n = {n}, norm(phi - prev_phi) = {err_phi}")
    print(f"n = {n}, norm(psi - prev_psi) = {err_psi}")
    n += 1
    prev_phi.assign(phi)
    prev_psi.assign(psi)

# we now calculate the map by which to move the mesh
f = Function(V, name="f")
f.interpolate(epsilon * ln(phi/mu))  

W = VectorFunctionSpace(mesh, "CG", 1)
proj_grad_f = Function(W, name="proj_grad_f")
proj_grad_f.project(grad(f)) 

U = FunctionSpace(mesh, "DG", 0)

mesh_map = Function(W)
mesh_map.interpolate(x + proj_grad_f)  # formula from McRae et al 2017

Vc = mesh.coordinates.function_space() 

# create the function which represents the mesh map
h = Function(Vc).interpolate(as_vector(mesh_map))

# we only want to move the interior points, so:
# make a point evaluator object with h and all the mesh points
point_evaluator = PointEvaluator(mesh, mesh.coordinates.dat.data)
h_at_points = point_evaluator.evaluate(h)

max_coords = [0.0, 0.0]
min_coords = [0.0, 0.0]

# keep 1s for the boundaries, move around the interior points
new_mesh_coords = [[1.0, 1.0] for _ in range(len(mesh.coordinates.dat.data))]

# moves the interior points by the optimal transport
for i in range(len(mesh.coordinates.dat.data[:,0])):
    x_co = mesh.coordinates.dat.data[i, 0] 
    y_co = mesh.coordinates.dat.data[i, 1]
    if (x_co < 33000000) and (x_co > -33000000) and (y_co < 33000000) and (y_co > -33000000):

        new_mesh_coords[i] = [h_at_points[i, 0], h_at_points[i, 1]]

        if x_co >= max_coords[0] and y_co >= max_coords[1]:
            max_coords = [x_co, y_co]

        if x_co <= min_coords[0] and y_co <= min_coords[1]:
            min_coords = [x_co, y_co]

# rescales the points back into the interior so there aren't
# large distorted cells between the boundary and the interior

interior_max = np.max(new_mesh_coords, axis=0)
interior_min = np.min(new_mesh_coords, axis=0)

for i in range(len(new_mesh_coords)):
    x_co = new_mesh_coords[i][0]
    y_co = new_mesh_coords[i][1]

    if (x_co != 1.0) and (y_co != 1.0):  # for non-boundary points
        # shifts the coords to fill the interior of the domain
        new_mesh_coords[i][0] = (x_co - interior_min[0])/(interior_max[0] - interior_min[0]) * 2 * max_coords[0] - max_coords[0]
        new_mesh_coords[i][1] = (y_co - interior_min[1])/(interior_max[1] - interior_min[1]) * 2 * max_coords[1] - max_coords[1]
        

# put OT coords into the mesh
for i in range(len(new_mesh_coords)):
    if new_mesh_coords[i] != [1.0, 1.0]:
        mesh.coordinates.dat.data[i] = new_mesh_coords[i] 

# to save the mesh created, uncomment the below lines:
#with CheckpointFile("new_mesh.h5", 'w') as afile:  # 'w' means create file
 #   afile.save_mesh(mesh)  # saves the new mesh to the file


