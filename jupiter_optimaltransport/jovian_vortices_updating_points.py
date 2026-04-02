import scipy
import numpy as np
import sympy as sp
import itertools

from gusto import *
from firedrake import (PeriodicRectangleMesh, SpatialCoordinate, as_vector,
                       interpolate, exp, sqrt, conditional)

dt = 500.          # timestep (in seconds) [originally 250]
tmax = 1730 * dt    # duration of the simulation (in seconds) [4320 * 500 = 25 days]
# ^ 10 days (technically 10.01157 days, not exact)

# Set up the mesh and choose the refinement level
Lx = 7e7
Ly = Lx 
nx = 256
ny = nx
mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
x, y = SpatialCoordinate(mesh)
mesh.coordinates.dat.data[:, 0] -= 0.5 * Lx  # moves the pole to the centre of the mesh
mesh.coordinates.dat.data[:, 1] -= 0.5 * Ly

# setup shallow water parameters
# specify g and Omega here if different from Earth values
H = 5e4 
g = 24.79
Omega = 1.74e-4
R = 71.4e6
parameters = ShallowWaterParameters(mesh, H=H, g=g, R=R, Omega=Omega,
                                    rotation=CoriolisOptions.gammaplane)

domain = Domain(mesh, dt, "RTCF", 1)


def smooth_f_profile(delta, rstar, Omega=parameters.Omega, R=R, Lx=Lx, nx=nx):
    """
    Returns the coefficients of a 5th order polynomial to smooth the transition from a polar gamma plane to the trap.

    Args:
      delta: half-width of smoothed region in number of cells
      rstar: radius of edge of trap in x-y grid coordinates
    """
    
    delta *= Lx/nx
    Omega = Omega.dat.data[0]
    r = sp.symbols('r')

    fexpr = 2*Omega*(1-0.5*r**2/R**2)
    left_val = fexpr.subs(r, rstar-delta)
    right_val = 2*Omega
    left_diff_val = sp.diff(fexpr, r).subs(r, rstar-delta)
    left_diff2_val = sp.diff(fexpr, r, 2).subs(r, rstar-delta)

    a = sp.symbols('a_0:6')
    P = a[0]
    for i in range(1, 6):
        P += a[i]*r**i
    eqns = [
        P.subs(r, rstar-delta) - left_val,
        P.subs(r, rstar+delta) - right_val,
        sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
        sp.diff(P, r).subs(r, rstar+delta),
        sp.diff(P, r, 2).subs(r, rstar-delta) - left_diff2_val,
        sp.diff(P, r, 2).subs(r, rstar+delta)
    ]

    sol = sp.solve(eqns, a)
    coeffs = [sol[sp.Symbol(f'a_{i}')] for i in range(6)]
    return coeffs


r, theta = rtheta_from_xy(x, y)

# define the expression for the gamma plane
fexpr = 2*Omega*(1-0.5*r**2/R**2)
# extract the coefficients of the smooth region. Here the trap radius is 3 cells from the edge of the domain, and the smoothing takes 2 cells either side of this trap radius
delta = 2
rstar = Lx/2 - 3*Lx/nx
coeffs = smooth_f_profile(delta, rstar)
# define a firedrake function using the polynomial coefficients
fsmooth = float(coeffs[0]) + float(coeffs[1])*r + float(coeffs[2])*r**2 + float(coeffs[3])*r**3 + float(coeffs[4])*r**4 + float(coeffs[5])*r**5
# use the gamma plane expression (fexpr), the smoothing (fsmooth) and the trap value (2*Omega) to build a full Coriolis expression
ftrap1 = conditional(r < rstar-delta*Lx/nx, fexpr, fsmooth)
ftrap = conditional(r < rstar+delta*Lx/nx, ftrap1, 2*Omega)

eqns = ShallowWaterEquations(domain, parameters)

# 15.72846 cores = use 16 cores
#print(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} ')


# field_creator object contains all the fields
def update_points(fields):

    D = fields("D") 
   # min_kernel = MinKernel()
    #D_min = min_kernel.apply(D)  # value of D min 

    # index which has the min of the np array for D
    idx = np.argmin(D.dat.data_ro)  
    V = D.function_space()
    m = V.mesh() 
    W = VectorFunctionSpace(m, V.ufl_element())
    X = assemble(interpolate(m.coordinates, W))  # puts the mesh coords into the vector function space

    min_point = X.dat.data_ro[idx]
  #  print(f"min_point = {min_point}")  #  [-57784, 18651534]

    # 1001 so there is a midpoint at index 500
    points_x = np.linspace(-1e7 + round(min_point[0]), 1e7 + round(min_point[0]), 1001)
    points_y = [round(min_point[1])]

    points = np.array([p for p in itertools.product(points_x, points_y)])
    
    return points #, min_point   - to track the vortex we could store the x_m,y_m values


dirname = "jv_up_D_10days"
dumpfreq = 173  # should give 11 outputs
pddumpfreq = dumpfreq 

output = OutputParameters(
    dirname=dirname, dumpfreq=dumpfreq, pddumpfreq=dumpfreq,
    dump_vtus=True, dump_nc=True,
    #point_data=[('PotentialVorticity', points)]  # field name and list of points at which to ouput this field
    point_data=[('D', update_points)]  # in the future want to include function that finds the new points too
)



# old output variable
#output = OutputParameters(dirname="jovian_vortices111125", dumpfreq=96)  # dumpfreq is how many iterations it does before it saves a state 


diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D'),
                     RelativeVorticity(method = 'project'), PotentialVorticity(method = 'project'),
                     ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(parameters),
                     ShallowWaterPotentialEnstrophy(),
                     CourantNumber()
                     ]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

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
#g = parameters.g.dat.data[0]
phi0 = g*H
Bu = phi0 / (f0 * rm)**2  # burger no. 

south_lat = np.deg2rad(south_lat_deg)
south_lon = np.deg2rad(south_lon_deg)

#print(f"south_lat = {south_lat}, south_lon = {south_lon}")


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


