from firedrake import *
from gusto import (GeneralIcosahedralSphereMesh, lonlatr_from_xyz,
                   rodrigues_rotation, lonlatr_components_from_xyz,
                   xyz_from_lonlatr, GeneralCubedSphereMesh)


R = 1 #71.4e6
radius = 71.4e6

# set up spherical mesh, originally is created with radius=1, 
# then is scaled up to full size after the mesh is distorted
# generalcubedmesh: 189 for nx128 equivalence, 95 for nx=64 equiv
mesh = GeneralCubedSphereMesh(R, num_cells_per_edge_of_panel=30)  
x = SpatialCoordinate(mesh)

# set up the functions to solve the OT between
# discontinuous circle at the pole
mu_expression = conditional(And((sqrt(x[0]**2 + x[1]**2) < 0.7), (x[2] > 0)), 
                Constant(1), Constant(0.00000000000001))
nu_expression = Constant(1.0)

# create the required function spaces on the mesh
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

# adjust mu and nu to have the same integral 
integral_value = R*R #* 10
mu.assign(mu * integral_value/integral_mu) 
nu.assign(nu * integral_value/integral_nu)

# initialise phi and psi to 1 
phi.interpolate(1)  
psi.interpolate(1)  

# set Sinkhorn iteration parameters
epsilon = 0.1
steps = 30
delta_t = epsilon/steps

print(f"epsilon = {epsilon}, delta_t = {delta_t}")

# for observing if the iterations are converging or not
err_phi = 1.0
prev_phi = Function(V)
prev_phi.assign(phi)

err_psi = 1.0
prev_psi = Function(V)
prev_psi.assign(psi)

n = 0

# the LHS and RHS of the PDE problem to solve the diffusion eqn
a = (Constant(delta_t) * inner(grad(u), grad(v)) + inner(u, v)) * dx 
L1 = inner(phi, v) * dx 
L2 = inner(psi, v) * dx

u = Function(V)  # for storing the solution of the pde


#while err_phi > 1e-6 or err_psi > 1e-6:
for _ in range(50):

    for i in range(steps):

        solve(a == L1, u)
        phi.assign(u)  # makes phi = next value of phi

    psi.interpolate(nu/phi)

    for i in range(steps):

        solve(a == L2, u)

        psi.assign(u)  

    phi.interpolate(mu/psi)

    # for testing convergence 
    err_phi = norm(phi - prev_phi) 
    err_psi = norm(psi - prev_psi)
    print(f"n = {n}, norm(phi - prev_phi) = {err_phi}")
    print(f"n = {n}, norm(psi - prev_psi) = {err_psi}")
    n += 1
    prev_phi.assign(phi)
    prev_psi.assign(psi)

# for creating the map to move the mesh
f = Function(V, name="f")
f.interpolate(epsilon * ln(phi/mu))  

W = VectorFunctionSpace(mesh, "CG", 1)

proj_grad_f = Function(W)
proj_grad_f.project(grad(f)) 

U = FunctionSpace(mesh, "DG", 0)

R_map = Function(W) 
lon, lat, r = lonlatr_from_xyz(x[0], x[1], x[2])

# turn proj_grad_f into lat lon r comps 
lon_f, lat_f, r_f = lonlatr_components_from_xyz(proj_grad_f, [x[0],x[1],x[2]])
new_lonlatr_coords = [lon+lon_f, lat+lat_f, r+r_f]  # apply the formula from McRae et al. 2017
new_x_expr, new_y_expr, new_z_expr = xyz_from_lonlatr(new_lonlatr_coords[0], new_lonlatr_coords[1], new_lonlatr_coords[2])

# create functions that represent how the mesh moves in each direction
lon_f_func = Function(V, name='lon_f').interpolate(lon_f)
lat_f_func = Function(V, name='lat_f').interpolate(lat_f)
r_f_func = Function(V, name='r_f').interpolate(r_f)

new_x = Function(V, name='new_x').interpolate(new_x_expr)
new_y = Function(V, name='new_y').interpolate(new_y_expr)
new_z = Function(V, name='new_z').interpolate(new_z_expr)

# calculates the new positions of the mesh points
point_evaluator = PointEvaluator(mesh, mesh.coordinates.dat.data)
new_x_coords = point_evaluator.evaluate(new_x)
new_y_coords = point_evaluator.evaluate(new_y)
new_z_coords = point_evaluator.evaluate(new_z)

Vc = mesh.coordinates.function_space()   

# puts the new mesh coords back into the mesh data
for i in range(len(mesh.coordinates.dat.data)):
    mesh.coordinates.dat.data[i] = [new_x_coords[i], new_y_coords[i], new_z_coords[i]]


# rescale to a sphere of correct full radius
g = Function(Vc).interpolate(as_vector([radius*x[0]/pow(x[0]**2 + x[1]**2 + x[2]**2, 0.5), radius*x[1]/pow(x[0]**2 + x[1]**2 + x[2]**2, 0.5), radius*x[2]/pow(x[0]**2 + x[1]**2 + x[2]**2, 0.5)]))
mesh.coordinates.assign(g)

# uncomment below lines to save the mesh file
#with CheckpointFile("mesh_cube_sphere_ot.h5", 'w') as afile:  # 'w' means create file
#    afile.save_mesh(mesh)  # saves the new mesh to the file