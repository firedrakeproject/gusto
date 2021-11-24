from os import path, makedirs
import itertools
from netCDF4 import Dataset
import sys
import time
from gusto.diagnostics import Diagnostics, Perturbation, SteadyStateError
from firedrake import (FiniteElement, TensorProductElement, HDiv,
                       FunctionSpace, MixedFunctionSpace, VectorFunctionSpace,
                       interval, Function, Mesh, functionspaceimpl,
                       File, SpatialCoordinate, sqrt, Constant, inner,
                       op2, DumbCheckpoint, FILE_CREATE, FILE_READ, interpolate,
                       CellNormal, cross, as_vector)
import numpy as np
from gusto.configuration import logger, set_log_handler

__all__ = ["State"]


class SpaceCreator(object):

    def __init__(self, mesh):
        self.mesh = mesh
        self.extruded_mesh = hasattr(mesh, "_base_mesh")
        self._initialised_base_spaces = False

    def __call__(self, name, family=None, degree=None):
        try:
            return getattr(self, name)
        except AttributeError:
            if name == "HDiv" and family in ["BDM", "RT", "CG"]:
                value = self.build_hdiv_space(family, degree)
            elif name == "theta":
                value = self.build_theta_space(degree)
            elif family == "DG":
                value = self.build_dg_space(degree)
            elif family == "CG":
                value = self.build_cg_space(degree)
            setattr(self, name, value)
            return value

    def build_compatible_spaces(self, family, degree):
        if self.extruded_mesh and not self._initialised_base_spaces:
            self.build_base_spaces(family, degree)
            Vu = self.build_hdiv_space(family, degree)
            setattr(self, "HDiv", Vu)
            Vdg = self.build_dg_space(degree)
            setattr(self, "DG", Vdg)
            Vth = self.build_theta_space(degree)
            setattr(self, "theta", Vth)
            return Vu, Vdg, Vth
        else:
            Vu = self.build_hdiv_space(family, degree)
            setattr(self, "HDiv", Vu)
            Vdg = self.build_dg_space(degree)
            setattr(self, "DG", Vdg)
            return Vu, Vdg

    def build_base_spaces(self, family, degree):

        cell = self.mesh._base_mesh.ufl_cell().cellname()

        # horizontal base spaces
        self.S1 = FiniteElement(family, cell, degree+1, variant="equispaced")
        self.S2 = FiniteElement("DG", cell, degree, variant="equispaced")

        # vertical base spaces
        self.T0 = FiniteElement("CG", interval, degree+1, variant="equispaced")
        self.T1 = FiniteElement("DG", interval, degree, variant="equispaced")

        self._initialised_base_spaces = True

    def build_hdiv_space(self, family, degree):
        if self.extruded_mesh:
            if not self._initialised_base_spaces:
                self.build_base_spaces(family, degree)
            Vh_elt = HDiv(TensorProductElement(self.S1, self.T1))
            Vt_elt = TensorProductElement(self.S2, self.T0)
            Vv_elt = HDiv(Vt_elt)
            V_elt = Vh_elt + Vv_elt
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement(family, cell, degree+1)
        return FunctionSpace(self.mesh, V_elt)

    def build_dg_space(self, degree):
        if self.extruded_mesh:
            if not self._initialised_base_spaces or self.T1.degree() != degree:
                cell = self.mesh._base_mesh.ufl_cell().cellname()
                S2 = FiniteElement("DG", cell, degree, variant="equispaced")
                T1 = FiniteElement("DG", interval, degree, variant="equispaced")
            else:
                S2 = self.S2
                T1 = self.T1
            V_elt = TensorProductElement(S2, T1)
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement("DG", cell, degree, variant="equispaced")
        return FunctionSpace(self.mesh, V_elt)

    def build_theta_space(self, degree):
        assert self.extruded_mesh
        if not self._initialised_base_spaces:
            cell = self.mesh._base_mesh.ufl_cell().cellname()
            self.S2 = FiniteElement("DG", cell, degree, variant="equispaced")
            self.T0 = FiniteElement("CG", interval, degree+1,
                                    variant="equispaced")
        V_elt = TensorProductElement(self.S2, self.T0)
        return FunctionSpace(self.mesh, V_elt)

    def build_cg_space(self, degree):
        return FunctionSpace(self.mesh, "CG", degree)


class FieldCreator(object):

    def __init__(self, equations):
        self.fields = []
        for eqn in equations:
            subfield_names = eqn.field_names if hasattr(eqn, "field_names") else None
            self.add_field(eqn.field_name, eqn.function_space, subfield_names)

    def add_field(self, name, space, subfield_names=None):
        value = Function(space, name=name)
        setattr(self, name, value)
        self.fields.append(value)

        if len(space) > 1:
            assert len(space) == len(subfield_names)
            for field_name, field in zip(subfield_names, value.split()):
                setattr(self, field_name, field)
                field.rename(field_name)
                self.fields.append(field)

    def __call__(self, name):
        return getattr(self, name)

    def __iter__(self):
        return iter(self.fields)


class StateFields(FieldCreator):

    def __init__(self, *fields_to_dump):
        self.fields = []
        self.output_specified = len(fields_to_dump) > 0
        self.to_dump = set((fields_to_dump))
        self.to_pickup = set(())

    def __call__(self, name, space=None, subfield_names=None, dump=True,
                 pickup=False):
        try:
            return getattr(self, name)
        except AttributeError:
            self.add_field(name, space, subfield_names)
            if dump:
                if subfield_names is not None:
                    self.to_dump.update(subfield_names)
                else:
                    self.to_dump.add(name)
            if pickup:
                self.to_pickup.add(name)
            return getattr(self, name)


class PointDataOutput(object):
    def __init__(self, filename, ndt, field_points, description,
                 field_creator, comm, tolerance=None, create=True):
        """Create a dump file that stores fields evaluated at points.

        :arg filename: The filename.
        :arg field_points: Iterable of pairs (field_name, evaluation_points).
        :arg description: Description of the simulation.
        :arg field_creator: The field creator (only used to determine
            datatype and shape of fields).
        :kwarg create: If False, assume that filename already exists
        """
        # Overwrite on creation.
        self.dump_count = 0
        self.filename = filename
        self.field_points = field_points
        self.tolerance = tolerance
        self.comm = comm
        if not create:
            return
        if self.comm.rank == 0:
            with Dataset(filename, "w") as dataset:
                dataset.description = "Point data for simulation {desc}".format(desc=description)
                dataset.history = "Created {t}".format(t=time.ctime())
                # FIXME add versioning information.
                dataset.source = "Output from Gusto model"
                # Appendable dimension, timesteps in the model
                dataset.createDimension("time", None)

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

                    # Get the UFL shape of the field
                    field_shape = field_creator(field_name).ufl_shape
                    # Number of geometric dimension occurences should be the same as the length of the UFL shape
                    field_len = len(field_shape)
                    field_count = field_shape.count(dim)
                    assert field_len == field_count, "Geometric dimension occurrences do not match UFL shape"
                    # Create the variable with the required shape
                    dimensions = ("time", "points") + field_count*("geometric_dimension",)
                    group.createVariable(field_name, field_creator(field_name).dat.dtype, dimensions)

    def dump(self, field_creator, t):
        """Evaluate and dump field data at points.

        :arg field_creator: :class:`FieldCreator` for accessing
            fields.
        :arg t: Simulation time at which dump occurs.
        """

        val_list = []
        for field_name, points in self.field_points:
            val_list.append((field_name, np.asarray(field_creator(field_name).at(points, tolerance=self.tolerance))))

        if self.comm.rank == 0:
            with Dataset(self.filename, "a") as dataset:
                # Add new time index
                dataset.variables["time"][self.dump_count] = t
                for field_name, vals in val_list:
                    group = dataset.groups[field_name]
                    var = group.variables[field_name]
                    var[self.dump_count, :] = vals

        self.dump_count += 1


class DiagnosticsOutput(object):
    def __init__(self, filename, diagnostics, description, comm, create=True):
        """Create a dump file that stores diagnostics.

        :arg filename: The filename.
        :arg diagnostics: The :class:`Diagnostics` object.
        :arg description: A description.
        :kwarg create: If False, assume that filename already exists
        """
        self.filename = filename
        self.diagnostics = diagnostics
        self.comm = comm
        if not create:
            return
        if self.comm.rank == 0:
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

        diagnostics = []
        for fname in self.diagnostics.fields:
            field = state.fields(fname)
            for dname in self.diagnostics.available_diagnostics:
                diagnostic = getattr(self.diagnostics, dname)
                diagnostics.append((fname, dname, diagnostic(field)))

        if self.comm.rank == 0:
            with Dataset(self.filename, "a") as dataset:
                idx = dataset.dimensions["time"].size
                dataset.variables["time"][idx:idx + 1] = t
                for fname, dname, value in diagnostics:
                    group = dataset.groups[fname]
                    var = group.variables[dname]
                    var[idx:idx + 1] = value


class State(object):
    """
    Build a model state to keep the variables in, and specify parameters.

    :arg mesh: The :class:`Mesh` to use.
    :arg sponge_function: (optional) Function specifying a sponge layer.
    :arg output: class containing output parameters
    :arg parameters: class containing physical parameters
    :arg diagnostics: class containing diagnostic methods
    :arg diagnostic_fields: list of diagnostic field classes
    """

    def __init__(self, mesh,
                 dt=None,
                 hydrostatic=None,
                 output=None,
                 parameters=None,
                 diagnostics=None,
                 diagnostic_fields=None):

        self.dt = dt
        self.hydrostatic = hydrostatic
        if output is None:
            raise RuntimeError("You must provide a directory name for dumping results")
        else:
            self.output = output
        self.parameters = parameters

        if diagnostics is not None:
            self.diagnostics = diagnostics
        else:
            self.diagnostics = Diagnostics()
        if diagnostic_fields is not None:
            self.diagnostic_fields = diagnostic_fields
        else:
            self.diagnostic_fields = []

        # The mesh
        self.mesh = mesh

        self.spaces = SpaceCreator(mesh)

        if self.output.dumplist is None:
            self.output.dumplist = []

        self.fields = StateFields(*self.output.dumplist)

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

        fields = set([f.name() for f in self.fields])
        field_deps = [(d, sorted(set(d.required_fields).difference(fields),)) for d in self.diagnostic_fields]
        schedule = topo_sort(field_deps)
        self.diagnostic_fields = schedule
        for diagnostic in self.diagnostic_fields:
            diagnostic.setup(self)
            self.diagnostics.register(diagnostic.name)

    def setup_dump(self, t, tmax, pickup=False):
        """
        Setup dump files
        Check for existence of directory so as not to overwrite
        output files
        Setup checkpoint file

        :arg tmax: model stop time
        :arg pickup: recover state from the checkpointing file if true,
        otherwise dump and checkpoint to disk. (default is False).
        """

        if any([self.output.dump_vtus, self.output.dumplist_latlon,
                self.output.dump_diagnostics, self.output.point_data,
                self.output.checkpoint and not pickup]):
            # setup output directory and check that it does not already exist
            self.dumpdir = path.join("results", self.output.dirname)
            running_tests = '--running-tests' in sys.argv or "pytest" in self.output.dirname
            if self.mesh.comm.rank == 0:
                if not running_tests and path.exists(self.dumpdir) and not pickup:
                    raise IOError("results directory '%s' already exists"
                                  % self.dumpdir)
                else:
                    if not running_tests:
                        makedirs(self.dumpdir)

        if self.output.dump_vtus:

            # setup pvd output file
            outfile = path.join(self.dumpdir, "field_output.pvd")
            self.dumpfile = File(
                outfile, project_output=self.output.project_fields,
                comm=self.mesh.comm)

            # make list of fields to dump
            self.to_dump = [f for f in self.fields if f.name() in self.fields.to_dump]

            # make dump counter
            self.dumpcount = itertools.count()

        # if there are fields to be dumped in latlon coordinates,
        # setup the latlon coordinate mesh and make output file
        if len(self.output.dumplist_latlon) > 0:
            mesh_ll = get_latlon_mesh(self.mesh)
            outfile_ll = path.join(self.dumpdir, "field_output_latlon.pvd")
            self.dumpfile_ll = File(outfile_ll,
                                    project_output=self.output.project_fields,
                                    comm=self.mesh.comm)

            # make functions on latlon mesh, as specified by dumplist_latlon
            self.to_dump_latlon = []
            for name in self.output.dumplist_latlon:
                f = self.fields(name)
                field = Function(
                    functionspaceimpl.WithGeometry(
                        f.function_space(), mesh_ll),
                    val=f.topological, name=name+'_ll')
                self.to_dump_latlon.append(field)

        # we create new netcdf files to write to, unless pickup=True, in
        # which case we just need the filenames
        if self.output.dump_diagnostics:
            diagnostics_filename = self.dumpdir+"/diagnostics.nc"
            self.diagnostic_output = DiagnosticsOutput(diagnostics_filename,
                                                       self.diagnostics,
                                                       self.output.dirname,
                                                       self.mesh.comm,
                                                       create=not pickup)

        if len(self.output.point_data) > 0:
            # set up point data output
            pointdata_filename = self.dumpdir+"/point_data.nc"
            ndt = int(tmax/self.dt)
            self.pointdata_output = PointDataOutput(pointdata_filename, ndt,
                                                    self.output.point_data,
                                                    self.output.dirname,
                                                    self.fields,
                                                    self.mesh.comm,
                                                    self.output.tolerance,
                                                    create=not pickup)

            # make point data dump counter
            self.pddumpcount = itertools.count()

            # set frequency of point data output - defaults to
            # dumpfreq if not set by user
            if self.output.pddumpfreq is None:
                self.output.pddumpfreq = self.output.dumpfreq

        # if we want to checkpoint and are not picking up from a previous
        # checkpoint file, setup the dumb checkpointing
        if self.output.checkpoint and not pickup:
            self.chkpt = DumbCheckpoint(path.join(self.dumpdir, "chkpt"),
                                        mode=FILE_CREATE)
            # make list of fields to pickup (this doesn't include
            # diagnostic fields)
            self.to_pickup = [f for f in self.fields if f.name() in self.fields.to_pickup]

        # if we want to checkpoint then make a checkpoint counter
        if self.output.checkpoint:
            self.chkptcount = itertools.count()

        # dump initial fields
        self.dump(t)

    def pickup_from_checkpoint(self):
        """
        :arg t: the current model time (default is zero).
        """
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
            raise ValueError("Must set checkpoint True if pickup")

        return t

    def dump(self, t):
        """
        Dump output
        """
        output = self.output

        # Diagnostics:
        # Compute diagnostic fields
        for field in self.diagnostic_fields:
            field(self)

        if output.dump_diagnostics:
            # Output diagnostic data
            self.diagnostic_output.dump(self, t)

        if len(output.point_data) > 0 and (next(self.pddumpcount) % output.pddumpfreq) == 0:
            # Output pointwise data
            self.pointdata_output.dump(self.fields, t)

        # Dump all the fields to the checkpointing file (backup version)
        if output.checkpoint and (next(self.chkptcount) % output.chkptfreq) == 0:
            for field in self.to_pickup:
                self.chkpt.store(field)
            self.chkpt.write_attribute("/", "time", t)

        if output.dump_vtus and (next(self.dumpcount) % output.dumpfreq) == 0:
            # dump fields
            self.dumpfile.write(*self.to_dump)

            # dump fields on latlon mesh
            if len(output.dumplist_latlon) > 0:
                self.dumpfile_ll.write(*self.to_dump_latlon)

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
            ref = self.fields(name+'bar')
            ref.interpolate(profile)

    def _build_spaces(self, mesh, vertical_degree, horizontal_degree, family):
        """
        Build:
        velocity space self.V2,
        pressure space self.V3,
        temperature space self.Vt,
        mixed function space self.W = (V2,V3,Vt)
        """

        if vertical_degree is not None:
            # horizontal base spaces
            cell = mesh._base_mesh.ufl_cell().cellname()
            S1 = FiniteElement(family, cell, horizontal_degree+1)
            S2 = FiniteElement("DG", cell, horizontal_degree, variant="equispaced")

            # vertical base spaces
            T0 = FiniteElement("CG", interval, vertical_degree+1, variant="equispaced")
            T1 = FiniteElement("DG", interval, vertical_degree, variant="equispaced")

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

            DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
            DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
            DG1_elt = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
            self.DG1_space = self.spaces("DG1", mesh, DG1_elt)

            self.W = MixedFunctionSpace((V0, V1, V2))

        else:
            cell = mesh.ufl_cell().cellname()
            V1_elt = FiniteElement(family, cell, horizontal_degree+1)
            DG_elt = FiniteElement("DG", cell, horizontal_degree, variant="equispaced")
            DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")

            V0 = self.spaces("HDiv", mesh, V1_elt)
            V1 = self.spaces("DG", mesh, DG_elt)
            self.DG1_space = self.spaces("DG1", mesh, DG1_elt)

            self.W = MixedFunctionSpace((V0, V1))


def get_latlon_mesh(mesh):
    coords_orig = mesh.coordinates
    coords_fs = coords_orig.function_space()

    if coords_fs.extruded:
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_elt = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
    else:
        cell = mesh.ufl_cell().cellname()
        DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)
    coords_dg = Function(vec_DG1).interpolate(coords_orig)
    coords_latlon = Function(vec_DG1)
    shapes = {"nDOFs": vec_DG1.finat_element.space_dimension(), 'dim': 3}

    radius = np.min(np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    # lat-lon 'x' = atan2(y, x)
    coords_latlon.dat.data[:, 0] = np.arctan2(coords_dg.dat.data[:, 1], coords_dg.dat.data[:, 0])
    # lat-lon 'y' = asin(z/sqrt(x^2 + y^2 + z^2))
    coords_latlon.dat.data[:, 1] = np.arcsin(coords_dg.dat.data[:, 2]/np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    # our vertical coordinate is radius - the minimum radius
    coords_latlon.dat.data[:, 2] = np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2) - radius

# We need to ensure that all points in a cell are on the same side of the branch cut in longitude coords
# This kernel amends the longitude coords so that all longitudes in one cell are close together
    kernel = op2.Kernel("""
#define PI 3.141592653589793
#define TWO_PI 6.283185307179586
void splat_coords(double *coords) {{
    double max_diff = 0.0;
    double diff = 0.0;

    for (int i=0; i<{nDOFs}; i++) {{
        for (int j=0; j<{nDOFs}; j++) {{
            diff = coords[i*{dim}] - coords[j*{dim}];
            if (fabs(diff) > max_diff) {{
                max_diff = diff;
            }}
        }}
    }}

    if (max_diff > PI) {{
        for (int i=0; i<{nDOFs}; i++) {{
            if (coords[i*{dim}] < 0) {{
                coords[i*{dim}] += TWO_PI;
            }}
        }}
    }}
}}
""".format(**shapes), "splat_coords")

    op2.par_loop(kernel, coords_latlon.cell_set,
                 coords_latlon.dat(op2.RW, coords_latlon.cell_node_map()))
    return Mesh(coords_latlon)


def topo_sort(field_deps):
    name2field = dict((f.name, f) for f, _ in field_deps)
    # map node: (input_deps, output_deps)
    graph = dict((f.name, (list(deps), [])) for f, deps in field_deps)
    roots = []
    for f, input_deps in field_deps:
        if len(input_deps) == 0:
            # No dependencies, candidate for evaluation
            roots.append(f.name)
        for d in input_deps:
            # add f as output dependency
            graph[d][1].append(f.name)

    schedule = []
    while roots:
        n = roots.pop()
        schedule.append(n)
        output_deps = list(graph[n][1])
        for m in output_deps:
            # Remove edge
            graph[m][0].remove(n)
            graph[n][1].remove(m)
            # If m now as no input deps, candidate for evaluation
            if len(graph[m][0]) == 0:
                roots.append(m)
    if any(len(i) for i, _ in graph.values()):
        cycle = "\n".join("%s -> %s" % (f, i) for f, (i, _) in graph.items()
                          if f not in schedule)
        raise RuntimeError("Field dependencies have a cycle:\n\n%s" % cycle)
    return list(map(name2field.__getitem__, schedule))
