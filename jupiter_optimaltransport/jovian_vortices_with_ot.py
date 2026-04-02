import scipy
import numpy as np
 
from gusto import *
from firedrake import (PeriodicRectangleMesh, CheckpointFile, 
                       SpatialCoordinate, as_vector,
                       interpolate, exp, sqrt, conditional)


dt = 500.          # timestep (in seconds) [originally 250]
tmax = 17280 * dt  # duration of the simulation (in seconds)


# load in the new ot adapted mesh:
with CheckpointFile("ot_128_centre.h5", 'r') as afile:
    mesh = afile.load_mesh()
x, y = SpatialCoordinate(mesh)


# setup shallow water parameters
# specify g and Omega here if different from Earth values
H = 5e4 
g = 24.79
Omega = 1.74e-4
R = 71.4e6

# set up teh shallow water parameters and equations
parameters = ShallowWaterParameters(mesh, H=H, g=g, R=R, Omega=Omega,
                                    rotation=CoriolisOptions.gammaplane)
domain = Domain(mesh, dt, "RTCF", 1)
eqns = ShallowWaterEquations(domain, parameters)

dirname = "jv_ot_square"
dumpfreq = 1728  # dumpfreq is how many iterations it does before it saves a state 

output = OutputParameters(dirname=dirname, dump_vtus=True, dumpfreq=dumpfreq)  

diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D'),
                     RelativeVorticity(method = 'project'), PotentialVorticity(method = 'project'),
                     ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(parameters),
                     ShallowWaterPotentialEnstrophy(),
                     CourantNumber()
                     ]

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
south_lon_deg = [0.]

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
    lamda_c = south_lon[idx]
    phi_c = south_lat[idx]

    # find the x-y (grid) coordinates of the centre of the vortex
    r_c, theta_c = rtheta_from_lonlat(lamda_c, phi_c, R)
    x_c, y_c = xy_from_rtheta(r_c, theta_c)

    # make an empty list of D values to append to
    D_values = []

    # loop over X coordinate values
    for Xval in X:
        x, y = Xval
        # calculate distance from centre of vortex to point
        dr = sqrt((x-x_c)**2 + (y-y_c)**2)
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

# calculate the initial velocity field
for i in range(len(south_lat)):
    r_c, theta_c = rtheta_from_lonlat(south_lon[i], south_lat[i], R)
    x_c, y_c = xy_from_rtheta(r_c, theta_c)
    dx = x - x_c
    dy = y - y_c
    dr = sqrt(dx**2 + dy**2)

    # Overide u,v components in velocity field
    mag_veloc = vm * (dr / rm) * exp((1/b) * (1 - (dr / rm)**b))

    u_veloc += - mag_veloc * (dy / dr)
    v_veloc += mag_veloc * (dx / dr)

uexpr = as_vector([u_veloc, v_veloc])

u0.project(uexpr)
D0.interpolate(Dfinal)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# Run the timestepper and generate the output.
stepper.run(t=0, tmax=tmax)
