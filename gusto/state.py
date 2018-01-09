from os import path
import itertools
from netCDF4 import Dataset
import time
from gusto.diagnostics import Diagnostics, Perturbation, \
    SteadyStateError
from firedrake import FiniteElement, TensorProductElement, HDiv, \
    FunctionSpace, MixedFunctionSpace, VectorFunctionSpace, \
    interval, Function, Mesh, functionspaceimpl,\
    File, SpatialCoordinate, sqrt, Constant, inner, \
    dx, op2, par_loop, READ, WRITE, DumbCheckpoint, \
    FILE_CREATE, FILE_READ, interpolate, CellNormal, cross, as_vector
import numpy as np
from gusto.configuration import logger, set_log_handler

__all__ = ["State"]


class SpaceCreator(object):

    def __call__(self, name, mesh=None, family=None, degree=None):
        try:
            return getattr(self, name)
        except AttributeError:
            value = FunctionSpace(mesh, family, degree)
            setattr(self, name, value)
            return value


class FieldCreator(object):

    def __init__(self, fieldlist=None, xn=None, dumplist=None, pickup=True):
        self.fields = []
        if fieldlist is not None:
            for name, func in zip(fieldlist, xn.split()):
                setattr(self, name, func)
                func.dump = name in dumplist
                func.pickup = pickup
                func.rename(name)
                self.fields.append(func)

    def __call__(self, name, space=None, dump=True, pickup=True):
        try:
            return getattr(self, name)
        except AttributeError:
            value = Function(space, name=name)
            setattr(self, name, value)
            value.dump = dump
            value.pickup = pickup
            self.fields.append(value)
            return value

    def __iter__(self):
        return iter(self.fields)


class PointDataOutput(object):
    def __init__(self, filename, ndt, field_points, description,
                 field_creator, create=True):
        """Create a dump file that stores fields evaluated at points.

        :arg filename: The filename.
        :arg field_points: Iterable of pairs (field_name, evaluation_points).
        :arg description: Description of the simulation.
        :arg field_creator: The field creator (only used to determine
            datatype of fields).
        :kwarg create: If False, assume that filename already exists
        """
        # Overwrite on creation.
        self.dump_count = 0
        self.filename = filename
        self.field_points = field_points
        if not create:
            return
        with Dataset(filename, "w") as dataset:
            dataset.description = "Point data for simulation {desc}".format(desc=description)
            dataset.history = "Created {t}".format(t=time.ctime())
            # FIXME add versioning information.
            dataset.source = "Output from Gusto model"
            # Appendable dimension, timesteps in the model
            dataset.createDimension("time", ndt+1)

            var = dataset.createVariable("time", np.float64, ("time"))
            var.units = "seconds"
            # Now create the variable group for each field
            for field_name, points in field_points:
                group = dataset.createGroup(field_name)
                npts, dim = points.shape
                group.createDimension("points", npts)
                group.createDimension("geometric_dimension", dim)
                var = group.createVariable("points", points.dtype,
                                           ("points", "geometric_dimension"))
                var[:] = points
                group.createVariable(field_name,
                                     field_creator(field_name).dat.dtype,
                                     ("time", "points"))

    def dump(self, field_creator, t):
        """Evaluate and dump field data at points.

        :arg field_creator: :class:`FieldCreator` for accessing
            fields.
        :arg t: Simulation time at which dump occurs.
        """
        with Dataset(self.filename, "a") as dataset:
            # Add new time index
            dataset.variables["time"][self.dump_count] = t
            for field_name, points in self.field_points:
                vals = np.asarray(field_creator(field_name).at(points))
                group = dataset.groups[field_name]
                var = group.variables[field_name]
                var[self.dump_count, :] = vals
        self.dump_count += 1


class DiagnosticsOutput(object):
    def __init__(self, filename, diagnostics, description, create=True):
        """Create a dump file that stores diagnostics.

        :arg filename: The filename.
        :arg diagnostics: The :class:`Diagnostics` object.
        :arg description: A description.
        :kwarg create: If False, assume that filename already exists
        """
        self.filename = filename
        self.diagnostics = diagnostics
        if not create:
            return
        with Dataset(filename, "w") as dataset:
            dataset.description = "Diagnostics data for simulation {desc}".format(desc=description)
            dataset.history = "Created {t}".format(t=time.ctime())
            dataset.source = "Output from Gusto model"
            dataset.createDimension("time", None)
            var = dataset.createVariable("time", np.float64, ("time", ))
            var.units = "seconds"
            for name in diagnostics.fields:
                group = dataset.createGroup(name)
                for diagnostic in diagnostics.available_diagnostics:
                    group.createVariable(diagnostic, np.float64, ("time", ))

    def dump(self, state, t):
        """Dump diagnostics.

        :arg state: The :class:`State` at which to compute the diagnostic.
        :arg t: The current time.
        """
        with Dataset(self.filename, "a") as dataset:
            idx = dataset.dimensions["time"].size
            dataset.variables["time"][idx:idx + 1] = t
            for name in self.diagnostics.fields:
                field = state.fields(name)
                group = dataset.groups[name]
                for dname in self.diagnostics.available_diagnostics:
                    diagnostic = getattr(self.diagnostics, dname)
                    var = group.variables[dname]
                    var[idx:idx + 1] = diagnostic(field)


class State(object):
    """
    Build a model state to keep the variables in, and specify parameters.

    :arg mesh: The :class:`Mesh` to use.
    :arg vertical_degree: integer, required for vertically extruded meshes.
    Specifies the degree for the pressure space in the vertical
    (the degrees for other spaces are inferred). Defaults to None.
    :arg horizontal_degree: integer, the degree for spaces in the horizontal
    (specifies the degree for the pressure space, other spaces are inferred)
    defaults to 1.
    :arg family: string, specifies the velocity space family to use.
    Options:
    "RT": The Raviart-Thomas family (default, recommended for quads)
    "BDM": The BDM family
    "BDFM": The BDFM family
    :arg Coriolis: (optional) Coriolis function.
    :arg sponge_function: (optional) Function specifying a sponge layer.
    :arg timestepping: class containing timestepping parameters
    :arg output: class containing output parameters
    :arg parameters: class containing physical parameters
    :arg diagnostics: class containing diagnostic methods
    :arg fieldlist: list of prognostic field names
    :arg diagnostic_fields: list of diagnostic field classes
    """

    def __init__(self, mesh, vertical_degree=None, horizontal_degree=1,
                 family="RT",
                 Coriolis=None, sponge_function=None,
                 hydrostatic=None,
                 timestepping=None,
                 output=None,
                 parameters=None,
                 diagnostics=None,
                 fieldlist=None,
                 diagnostic_fields=None):

        self.Omega = Coriolis
        self.mu = sponge_function
        self.hydrostatic = hydrostatic
        self.timestepping = timestepping
        if output is None:
            raise RuntimeError("You must provide a directory name for dumping results")
        else:
            self.output = output
        self.parameters = parameters
        if fieldlist is None:
            raise RuntimeError("You must provide a fieldlist containing the names of the prognostic fields")
        else:
            self.fieldlist = fieldlist
        if diagnostics is not None:
            self.diagnostics = diagnostics
        else:
            self.diagnostics = Diagnostics(*fieldlist)
        if diagnostic_fields is not None:
            self.diagnostic_fields = diagnostic_fields
        else:
            self.diagnostic_fields = []

        # The mesh
        self.mesh = mesh

        # Build the spaces
        self._build_spaces(mesh, vertical_degree, horizontal_degree, family)

        # Allocate state
        self._allocate_state()
        if self.output.dumplist is None:
            self.output.dumplist = fieldlist
        self.fields = FieldCreator(fieldlist, self.xn, self.output.dumplist)

        self.dumpfile = None

        # figure out if we're on a sphere
        try:
            self.on_sphere = (mesh._base_mesh.geometric_dimension() == 3 and mesh._base_mesh.topological_dimension() == 2)
        except AttributeError:
            self.on_sphere = (mesh.geometric_dimension() == 3 and mesh.topological_dimension() == 2)

        #  build the vertical normal and define perp for 2d geometries
        dim = mesh.topological_dimension()
        if self.on_sphere:
            x = SpatialCoordinate(mesh)
            R = sqrt(inner(x, x))
            self.k = interpolate(x/R, mesh.coordinates.function_space())
            if dim == 2:
                outward_normals = CellNormal(mesh)
                self.perp = lambda u: cross(outward_normals, u)
        else:
            kvec = [0.0]*dim
            kvec[dim-1] = 1.0
            self.k = Constant(kvec)
            if dim == 2:
                self.perp = lambda u: as_vector([-u[1], u[0]])

        # project test function for hydrostatic case
        if self.hydrostatic:
            self.h_project = lambda u: u - self.k*inner(u, self.k)
        else:
            self.h_project = lambda u: u

        #  Constant to hold current time
        self.t = Constant(0.0)

        # setup logger
        logger.setLevel(output.log_level)
        set_log_handler(mesh.comm)
        logger.info("Timestepping parameters that take non-default values:")
        logger.info(", ".join("%s: %s" % item for item in vars(timestepping).items()))
        if parameters is not None:
            logger.info("Physical parameters that take non-default values:")
            logger.info(", ".join("%s: %s" % item for item in vars(parameters).items()))

    def setup_diagnostics(self):
        """
        Add special case diagnostic fields
        """
        for name in self.output.perturbation_fields:
            f = Perturbation(name)
            self.diagnostic_fields.append(f)

        for name in self.output.steady_state_error_fields:
            f = SteadyStateError(self, name)
            self.diagnostic_fields.append(f)

        for diagnostic in self.diagnostic_fields:
            diagnostic.setup(self)
            self.diagnostics.register(diagnostic.name)

    def setup_dump(self, tmax, pickup=False):
        """
        Setup dump files
        Check for existence of directory so as not to overwrite
        output files
        Setup checkpoint file

        :arg tmax: model stop time
        :arg pickup: recover state from the checkpointing file if true,
        otherwise dump and checkpoint to disk. (default is False).
        """
        self.dumpdir = path.join("results", self.output.dirname)
        outfile = path.join(self.dumpdir, "field_output.pvd")
        if self.mesh.comm.rank == 0 and "pytest" not in self.output.dirname \
           and path.exists(self.dumpdir) and not pickup:
            raise IOError("results directory '%s' already exists" % self.dumpdir)
        self.dumpcount = itertools.count()
        self.dumpfile = File(outfile, project_output=self.output.project_fields, comm=self.mesh.comm)
        if self.output.checkpoint and not pickup:
            self.chkpt = DumbCheckpoint(path.join(self.dumpdir, "chkpt"), mode=FILE_CREATE)

        # make list of fields to dump
        self.to_dump = [field for field in self.fields if field.dump]

        # if there are fields to be dumped in latlon coordinates,
        # setup the latlon coordinate mesh and make output file
        if len(self.output.dumplist_latlon) > 0:
            mesh_ll = get_latlon_mesh(self.mesh)
            outfile_ll = path.join(self.dumpdir, "field_output_latlon.pvd")
            self.dumpfile_ll = File(outfile_ll,
                                    project_output=self.output.project_fields,
                                    comm=self.mesh.comm)

        # make list of fields to pickup (this doesn't include diagnostic fields)
        self.to_pickup = [field for field in self.fields if field.pickup]

        # make functions on latlon mesh, as specified by dumplist_latlon
        self.to_dump_latlon = []
        for name in self.output.dumplist_latlon:
            f = self.fields(name)
            field = Function(functionspaceimpl.WithGeometry(f.function_space(), mesh_ll), val=f.topological, name=name+'_ll')
            self.to_dump_latlon.append(field)

        # we create new netcdf files to write to, unless pickup=True, in
        # which case we just need the filenames
        if self.output.dump_diagnostics:
            diagnostics_filename = self.dumpdir+"/diagnostics.nc"
            self.diagnostic_output = DiagnosticsOutput(diagnostics_filename,
                                                       self.diagnostics,
                                                       self.output.dirname,
                                                       create=not pickup)

        if len(self.output.point_data) > 0:
            pointdata_filename = self.dumpdir+"/point_data.nc"

            ndt = int(tmax/self.timestepping.dt)
            self.pointdata_output = PointDataOutput(pointdata_filename, ndt,
                                                    self.output.point_data,
                                                    self.output.dirname,
                                                    self.fields,
                                                    create=not pickup)

    def dump(self, t=0, pickup=False):
        """
        Dump output
        :arg t: the current model time (default is zero).
        :arg pickup: recover state from the checkpointing file if true,
        otherwise dump and checkpoint to disk. (default is False).
        """
        if pickup:
            if self.output.checkpoint:
                # Open the checkpointing file for writing
                chkfile = path.join(self.dumpdir, "chkpt")
                with DumbCheckpoint(chkfile, mode=FILE_READ) as chk:
                    # Recover all the fields from the checkpoint
                    for field in self.to_pickup:
                        chk.load(field)
                    t = chk.read_attribute("/", "time")
                    next(self.dumpcount)
                # Setup new checkpoint
                self.chkpt = DumbCheckpoint(path.join(self.dumpdir, "chkpt"), mode=FILE_CREATE)
            else:
                raise NotImplementedError("Must set checkpoint True if pickup")
        else:

            if self.output.dump_diagnostics:
                # Compute diagnostic fields
                for field in self.diagnostic_fields:
                    field(self)

                # Output diagnostic data
                self.diagnostic_output.dump(self, t)

            if len(self.output.point_data) > 0:
                # Output pointwise data
                self.pointdata_output.dump(self.fields, t)

            # Dump all the fields to the checkpointing file (backup version)
            if self.output.checkpoint:
                for field in self.to_pickup:
                    self.chkpt.store(field)
                self.chkpt.write_attribute("/", "time", t)

            if (next(self.dumpcount) % self.output.dumpfreq) == 0:
                # dump fields
                self.dumpfile.write(*self.to_dump)

                # dump fields on latlon mesh
                if len(self.output.dumplist_latlon) > 0:
                    self.dumpfile_ll.write(*self.to_dump_latlon)

        return t

    def initialise(self, initial_conditions):
        """
        Initialise state variables

        :arg initial_conditions: An iterable of pairs (field_name, pointwise_value)
        """
        for name, ic in initial_conditions:
            f_init = getattr(self.fields, name)
            f_init.assign(ic)
            f_init.rename(name)

    def set_reference_profiles(self, reference_profiles):
        """
        Initialise reference profiles

        :arg reference_profiles: An iterable of pairs (field_name, interpolatory_value)
        """
        for name, profile in reference_profiles:
            field = getattr(self.fields, name)
            ref = self.fields(name+'bar', field.function_space(), False)
            ref.interpolate(profile)

    def _build_spaces(self, mesh, vertical_degree, horizontal_degree, family):
        """
        Build:
        velocity space self.V2,
        pressure space self.V3,
        temperature space self.Vt,
        mixed function space self.W = (V2,V3,Vt)
        """

        self.spaces = SpaceCreator()
        if vertical_degree is not None:
            # horizontal base spaces
            cell = mesh._base_mesh.ufl_cell().cellname()
            S1 = FiniteElement(family, cell, horizontal_degree+1)
            S2 = FiniteElement("DG", cell, horizontal_degree)

            # vertical base spaces
            T0 = FiniteElement("CG", interval, vertical_degree+1)
            T1 = FiniteElement("DG", interval, vertical_degree)

            # build spaces V2, V3, Vt
            V2h_elt = HDiv(TensorProductElement(S1, T1))
            V2t_elt = TensorProductElement(S2, T0)
            V3_elt = TensorProductElement(S2, T1)
            V2v_elt = HDiv(V2t_elt)
            V2_elt = V2h_elt + V2v_elt

            V0 = self.spaces("HDiv", mesh, V2_elt)
            V1 = self.spaces("DG", mesh, V3_elt)
            V2 = self.spaces("HDiv_v", mesh, V2t_elt)

            self.Vv = self.spaces("Vv", mesh, V2v_elt)

            self.W = MixedFunctionSpace((V0, V1, V2))

        else:
            cell = mesh.ufl_cell().cellname()
            V1_elt = FiniteElement(family, cell, horizontal_degree+1)

            V0 = self.spaces("HDiv", mesh, V1_elt)
            V1 = self.spaces("DG", mesh, "DG", horizontal_degree)

            self.W = MixedFunctionSpace((V0, V1))

    def _allocate_state(self):
        """
        Construct Functions to store the state variables.
        """

        W = self.W
        self.xn = Function(W)
        self.xstar = Function(W)
        self.xp = Function(W)
        self.xnp1 = Function(W)
        self.xrhs = Function(W)
        self.xb = Function(W)  # store the old state for diagnostics
        self.dy = Function(W)


def get_latlon_mesh(mesh):
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
    coords_latlon.dat.data[:, 0] = np.arctan2(coords_dg.dat.data[:, 1], coords_dg.dat.data[:, 0])
    # lat-lon 'y' = asin(z/sqrt(x^2 + y^2 + z^2))
    coords_latlon.dat.data[:, 1] = np.arcsin(coords_dg.dat.data[:, 2]/np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    coords_latlon.dat.data[:, 2] = 0.0

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
}
""", "splat_coords")

    op2.par_loop(kernel, coords_latlon.cell_set,
                 coords_latlon.dat(op2.RW, coords_latlon.cell_node_map()))
    return Mesh(coords_latlon)
