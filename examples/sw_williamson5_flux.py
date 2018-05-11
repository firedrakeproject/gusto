from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, \
    as_vector, pi, sqrt, Min, FunctionSpace, VectorFunctionSpace, \
    Function, par_loop, READ, WRITE, dx, Mesh, assemble, interpolate, \
    File, functionspaceimpl
from os import path
import sys
from netCDF4 import Dataset
import numpy as np
import json

def get_data(X, n, filename, xfilename):
    temp = np.fromfile(filename)
    Xref = np.fromfile(xfilename).reshape(-1, 3)
    result = []
    for x in X:
        ii = [np.allclose(x, Xref[i, :]) for i in range(Xref.shape[0])]
        j = np.where(ii)[0][0]
        result.append(temp[j])
    return np.array(result)

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 900.}
    tmax = 900.
    dumptime = 900.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 900., 4: 450., 5: 225., 6: 112.5, 6:84.375}
    tmax = 15.*day
    dumptime = 5*60.*60.

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    dumpfreq = int(dumptime/dt)
    dirname = "sw_W5_50days_fluxform_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname, dumplist=['D', 'u', 'PotentialVorticity'], dumplist_latlon=['D', 'PotentialVorticity', 'VelocityDivergence'], dumpfreq=dumpfreq)
    diagnostic_fields = [Sum('D', 'topography'),
                         CourantNumber(),
                         PotentialVorticity(),
                         ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(),
                         ShallowWaterPotentialEnstrophy(),
                         VelocityDivergence()]

    state = State(mesh, horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
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
    b0 = Function(V).interpolate(bexpr)
    b = state.fields("topography", D0.function_space())
    b.project(b0)

    Vvec = VectorFunctionSpace(mesh, "CG", 3)
    u = Function(Vvec).interpolate(uexpr)
    u0.project(u)
    D = Function(V).interpolate(Dexpr)
    D0.project(D)
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

    # Get reference solution
     refdir = "examples/williamson5_reference_solution/"
     Dref_filename = path.join(refdir, "Dref%s.dat" % ref_level)
     Xref_filename = path.join(refdir, "xref%s.dat" % ref_level)
     VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
     X = interpolate(mesh.coordinates, VectorCG1)
     CG1 = FunctionSpace(mesh, "CG", 1)
     Dref_cg = Function(CG1, name="Dref_cg")
     Dref_cg.dat.data[:] = get_data(X.dat.data_ro, ref_level, Dref_filename,
                                    Xref_filename)
     Dref = Function(D.function_space())
     err = Function(D.function_space())

    # copy reference solution
     par_loop("""
     for (int i=0; i<dg.dofs; i++) {
         dg[i][0] = cg[i][0];
     }
     """, dx, {'dg': (Dref, WRITE),
               'cg': (Dref_cg, READ)})

     err.assign(D - Dref)
    
    # Setup output files
    outdir = path.join("results", dirname)
    outfile = File(path.join(outdir, "err.pvd"))
    outfile_left = File(path.join(outdir, "err_left.pvd"))
    outfile_right = File(path.join(outdir, "err_right.pvd"))

     outfile.write(err, D, Dref, Dref_cg)

    # calculate and write out errors
    flist = ["Dref", "err"]
    diagnostics = Diagnostics(*flist)
    Dref_diagnostics = {}
    for dname in ["max", "l2"]:
        diagnostic = getattr(diagnostics, dname)
        Dref_diagnostics[dname] = diagnostic(Dref)
    error_diagnostics = {}
    for dname in ["max", "l2"]:
        diagnostic = getattr(diagnostics, dname)
        error_diagnostics[dname] = diagnostic(err)
    l2_err = error_diagnostics["l2"]/Dref_diagnostics["l2"]
    linf_err = error_diagnostics["max"]/Dref_diagnostics["max"]
    f = open(path.join(outdir, "errors"), "w")
    f.write("Dref:  ")
    f.write(json.dumps(Dref_diagnostics))
    f.write("\n")
    f.write("err:  ")
    f.write(json.dumps(error_diagnostics))
    f.write("\n")
    f.write("l2 D error: %s \n" % str(l2_err))
    f.write("l_inf D error: %s" % str(linf_err))
    f.close()

    ### GENERATE LAT-LON COORDINATES ###
    VectorDG1 = VectorFunctionSpace(mesh, "DG", 1)
    coords_dg = Function(VectorDG1)

    # copy coordinate field into DG
    par_loop("""
    for (int i=0; i<dg.dofs; i++) {
        for (int j=0; j<3; j++) {
             dg[i][j] = cg[i][j];
         }
     }
     """, dx, {'dg': (coords_dg, WRITE),
               'cg': (mesh.coordinates, READ)})

    DG1_2D = VectorFunctionSpace(mesh, "DG", 1, dim=2)
    coords_latlon = Function(DG1_2D)

    # lat-lon 'x' = atan2(y, x)
    coords_latlon.dat.data[:,0] = np.arctan2(coords_dg.dat.data[:,1], coords_dg.dat.data[:,0])
    # lat-lon 'y' = asin(z/sqrt(x^2 + y^2 + z^2))
    coords_latlon.dat.data[:,1] = np.arcsin(coords_dg.dat.data[:,2]/np.sqrt(coords_dg.dat.data[:,0]**2 + coords_dg.dat.data[:,1]**2 + coords_dg.dat.data[:,2]**2))

    coords_latlon_left = Function(DG1_2D)
    coords_latlon_right = Function(DG1_2D)

    par_loop("""
    #define PI 3.141592653589793
    #define TWO_PI 6.283185307179586

    // Copy fields
    for (int i=0; i<latlon.dofs; i++) {
        for (int j=0; j<2; j++) {
            left[i][j] = latlon[i][j];
            right[i][j] = latlon[i][j];
        }
    }

    double diff0 = (latlon[0][0] - latlon[1][0]);
    double diff1 = (latlon[0][0] - latlon[2][0]);

    // If cell is wrapped around...
    if (fabs(diff0) > PI || fabs(diff1) > PI) {
        const int sign0 = latlon[0][0] < 0 ? -1 : 1;
        const int sign1 = latlon[1][0] < 0 ? -1 : 1;
        const int sign2 = latlon[2][0] < 0 ? -1 : 1;

        if (sign0 < 0)
            right[0][0] += TWO_PI;
        else
            left[0][0] -= TWO_PI;

        if (sign1 < 0)
            right[1][0] += TWO_PI;
        else
            left[1][0] -= TWO_PI;

        if (sign2 < 0)
            right[2][0] += TWO_PI;
        else
            left[2][0] -= TWO_PI;

    }
    """, dx, {'latlon': (coords_latlon, READ),
              'left': (coords_latlon_left, WRITE),
              'right': (coords_latlon_right, WRITE)})
    ### END ###

    mesh_l = Mesh(coords_latlon_left)
    err_l = Function(functionspaceimpl.WithGeometry(err.function_space(), mesh_l), val=err.topological)
    outfile_left.write(err_l)

    mesh_r = Mesh(coords_latlon_right)
    err_r = Function(functionspaceimpl.WithGeometry(err.function_space(), mesh_r), val=err.topological)
    outfile_right.write(err_r)

