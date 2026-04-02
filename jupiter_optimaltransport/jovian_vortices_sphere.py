import scipy
import numpy as np

from gusto import *
from firedrake import (IcosahedralSphereMesh, CheckpointFile, 
                       SpatialCoordinate, as_vector,
                       interpolate, exp, sqrt, conditional, 
                       VTKFile, acos, cos, sin, dx, assemble,
                       PointEvaluator, CubedSphereMesh)

 
dt = 500.          # timestep (in seconds) 
tmax = 17280 * dt  # duration of the simulation (in seconds) 

# setup shallow water parameters
# specify g and Omega here if different from Earth values
H = 5e4 
g = 24.79
Omega = 1.74e-4
R = 71.4e6

# Set up the mesh and choose the refinement level
if False:  # set to True for a uniform mesh
    mesh = GeneralCubedSphereMesh(R, 95) 

if True:  # set to True to use the mesh created by OT
    with CheckpointFile("7_0_1_cube_sphere_ot.h5", 'r') as afile:
        mesh = afile.load_mesh()

x, y, z = SpatialCoordinate(mesh)

# create parameters object
parameters = ShallowWaterParameters(mesh, H=H, g=g, R=R, Omega=Omega,
                                    rotation=CoriolisOptions.sphere)

domain = Domain(mesh, dt, "RTCF", 1)

eqns = ShallowWaterEquations(domain, parameters)

dirname = "jv_OT_cube_sphere_100days"
dumpfreq = 1728  # dumpfreq is how many iterations it does before it saves a state 


output = OutputParameters(dirname=dirname, dump_vtus=True, dumpfreq=dumpfreq)  

diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D'),
                     RelativeVorticity(method = 'solve'), PotentialVorticity(method = 'solve')]

io = IO(domain, output=output, diagnostic_fields=diagnostic_fields)

subcycling_options = SubcyclingOptions(subcycle_by_courant=0.33)
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D", subcycling_options=subcycling_options)]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods, reference_update_freq = 0)

u0 = stepper.fields("u")
D0 = stepper.fields("D")

# cyclone initial place 
south_lat_deg = [75.]  
south_lon_deg = [90.]  

b = 1.0               # Steepness parameter  - look in paper at diff values of b for diff amounts of shield
Ro = 0.23             # Rossby Number
f0 = 2 * Omega        # Planetary vorticity
rm = 1e6              # Radius of vortex (m)
vm = Ro * f0 * rm     # Calculate speed with Ro
phi0 = g*H
Bu = phi0 / (f0 * rm)**2  # burger no. 

south_lat = np.deg2rad(south_lat_deg)
south_lon = np.deg2rad(south_lon_deg)


def initialise_D(X, idx):
    # computes the initial depth perturbation corresponding to vortex
    # idx, given coordinates X

    # get the lon, lat coordinates of the centre of this vortex
    lon_c = south_lon[idx]
    lat_c = south_lat[idx]

    # find the xyz coordinates of the centre of the vortex
    x_c, y_c, z_c = xyz_from_lonlatr(lon_c, lat_c, R)

    # make an empty list of D values to append to
    D_values = []

    # loop over X coordinate values
    for Xval in X:
        x, y, z = Xval
        # calculate distance from centre of vortex to point
        lon2, lat2, _ = lonlatr_from_xyz(x, y, z)
        dr = great_arc_angle(lon_c, lat_c, lon2, lat2) * R
        phi_perturb = (Ro/Bu)*exp(1./b)*(b**(-1.+(2./b)))*scipy.special.gammaincc(2./b, (1./b)*(dr/rm)**b)*scipy.special.gamma(2./b)
        D_values.append(-1 * H * phi_perturb)

    # return list of D values in correct order
    return D_values


# get the function space from the field we would like to initialise (D0)
VD = D0.function_space()
# create a vector function space for the coordinate values with the
# same mesh and finite element as D0
W = VectorFunctionSpace(VD.mesh(), VD.ufl_element())
# set up a function, X, with the value of the coordinates
# corresponding to the DOFs of D0
X = assemble(interpolate(VD.mesh().coordinates, W))
# set up a temporary function with the same structure as D to hold the
# values from each vortex
Dtmp = Function(D0.function_space())
Dfinal = Function(D0.function_space())
# loop over vortices
for idx in range(len(south_lat)):
    # calculate depth perturbation for each vortex
    Dtmp.dat.data[:] = initialise_D(X.dat.data_ro, idx)
    # add on to D0
    Dfinal += Dtmp
Dfinal += H

u_veloc = 0.*x
v_veloc = 0.*y
w_veloc = 0.*z 

# create the velocity field over the sphere
for i in range(len(south_lat)):
    x_c, y_c, z_c = xyz_from_lonlatr(south_lon[i], south_lat[i], R)

    lon, lat, r = lonlatr_from_xyz(x, y, z)
    lon_c, lat_c, _ = lonlatr_components_from_xyz([x,y,z], [x_c, y_c, z_c])

    # creates the velocity using lonlatr coords
    dlon = (lon - lon_c)
    dlat = (lat - lat_c)
    lon_c, lat_c, _ = lonlatr_from_xyz(x_c, y_c, z_c)
    dr = great_arc_angle(lon, lat, lon_c, lat_c) * R 

    lon_veloc = - vm * (dlat / rm) * exp((1/b) * (1 - (dr / rm)**b)) 
    lat_veloc = + vm * (dlon / rm) * exp((1/b) * (1 - (dr / rm)**b)) 
    
    # changes back into xyz coords
    u_vect = xyz_vector_from_lonlatr(lon_veloc, lat_veloc, 0, [lon, lat, r])

    u_veloc += lon_veloc
    v_veloc += lat_veloc

uexpr = as_vector([u_veloc, v_veloc, w_veloc])

u0.project(uexpr)
D0.interpolate(Dfinal)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# Run the timestepper and generate the output.
stepper.run(t=0, tmax=tmax)