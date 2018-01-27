from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, \
    as_vector, pi, sqrt, Min, FunctionSpace
from os import path
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 900.}
    tmax = 900.
    dumptime = 900.
else:
    # setup resolution and timestepping parameters for convergence test
    # ref_dt = {3: 900., 4: 450., 5: 225., 6: 112.5}
    ref_dt = {4: 450.}
    tmax = 15*day
    dumptime = 5*60.*60.

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    dumpfreq = int(dumptime/dt)
    dirname = "sw_W5_fluxform_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname, dumplist=['D', 'u', 'PotentialVorticity'], dumplist_latlon=['D', 'PotentialVorticity'], dumpfreq=dumpfreq)
    diagnostic_fields = [Sum('D', 'topography'),
                         PotentialVorticity(),
                         ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(),
                         ShallowWaterPotentialEnstrophy(),]

    state = State(mesh, horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = state.fields('u')
    D0 = state.fields('D')
    x = SpatialCoordinate(mesh)
    u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    theta, lamda = latlon_coords(mesh)
    Omega = parameters.Omega
    g = parameters.g
    Rsq = R**2
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = Min(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

    # Coriolis
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 3)
    f = state.fields("coriolis", V)
    f.interpolate(fexpr)  # Coriolis frequency (1/s)
    b = state.fields("topography", D0.function_space())
    b.interpolate(bexpr)

    u0.project(uexpr)
    D0.project(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    mass_flux = MassFluxReconstruction(state)
    pv_flux = PVFluxTaylorGalerkin(state, mass_flux)

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, euler_poincare=False, mass_flux=mass_flux.flux, pv_flux=pv_flux.flux)

    # build time stepper
    stepper = FluxForm(state, linear_solver, sw_forcing, fluxes=[mass_flux, pv_flux])

    stepper.run(t=0, tmax=tmax)

    final_fields = stepper.state.fields
    D = final_fields("D_plus_topography")
    dname = path.join("results", dirname)
    fname = path.join(dname, "D.dat")
    D.dat.data.tofile(fname)
