from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, \
    FunctionSpace
from math import pi
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 3000.}  # 4: 1500., 5: 750., 6: 375.}
    tmax = 5.*day

# setup shallow water parameters
R = 6371220.
H = 2.94e4/9.80616

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    dirname = "new_fluxform_sw_W2_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    global_normal = x
    mesh.init_cell_orientations(x)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname, dumplist_latlon=['D', 'D_error'], steady_state_error_fields=['D', 'u'])
    diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                         ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(),
                         ShallowWaterPotentialEnstrophy()]

    state = State(mesh, horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    x = SpatialCoordinate(mesh)
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    Omega = parameters.Omega
    g = parameters.g
    Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
    # Coriolis expression
    fexpr = 2*Omega*x[2]/R
    V0 = FunctionSpace(mesh, "CG", 3)
    # V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", V0)
    f.interpolate(fexpr)  # Coriolis frequency (1/s)

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    state.initialise([('u', u0),
                      ('D', D0)])

    mass_flux = MassFluxReconstruction(state)
    pv_flux = PVFluxTaylorGalerkin(state, mass_flux)

    linear_solver = ShallowWaterSolver(state)

    forcing = ShallowWaterForcing(state, euler_poincare=False, mass_flux=mass_flux.flux, pv_flux=pv_flux.flux)

    # build time stepper
    stepper = FluxForm(state, linear_solver, forcing, fluxes=[mass_flux, pv_flux])

    stepper.run(t=0, tmax=tmax)
