"""Provides the model's IO, which controls input, output and diagnostics."""

from os import path, makedirs
import itertools
from netCDF4 import Dataset
import sys
import time
from gusto.diagnostics import Diagnostics
from gusto.meshes import get_flat_latlon_mesh
from firedrake import (Function, functionspaceimpl, File,
                       DumbCheckpoint, FILE_CREATE, FILE_READ, CheckpointFile)
import numpy as np
from gusto.logging import logger, update_logfile_location

__all__ = ["pick_up_mesh", "IO"]


def pick_up_mesh(output, mesh_name):
    """
    Picks up a checkpointed mesh. This must be the first step of any model being
    picked up from a checkpointing run.

    Args:
        output (:class:`OutputParameters`): holds and describes the options for
            outputting.
        mesh_name (str): the name of the mesh to be picked up. The default names
            used by Firedrake are "firedrake_default" for non-extruded meshes,
            or "firedrake_default_extruded" for extruded meshes.

    Returns:
        :class:`Mesh`: the mesh to be used by the model.
    """

    # Open the checkpointing file for writing
    dumpdir = None
    if output.checkpoint_pickup_filename is not None:
        chkfile = output.checkpoint_pickup_filename
    else:
        dumpdir = path.join("results", output.dirname)
        chkfile = path.join(dumpdir, "chkpt.h5")
    with CheckpointFile(chkfile, 'r') as chk:
        mesh = chk.load_mesh(mesh_name)

    if dumpdir:
        update_logfile_location(dumpdir, mesh.comm)

    return mesh


class PointDataOutput(object):
    """Object for outputting field point data."""
    def __init__(self, filename, field_points, description,
                 field_creator, comm, tolerance=None, create=True):
        """
        Args:
            filename (str): name of file to output to.
            field_points (list): some iterable of pairs, matching fields with
                arrays of evaluation points: (field_name, evaluation_points).
            description (str): a description of the simulation to be included in
                the output.
            field_creator (:class:`FieldCreator`): the field creator, used to
                determine the datatype and shape of fields.
            comm (:class:`MPI.Comm`): MPI communicator.
            tolerance (float, optional): tolerance to use for the evaluation of
                fields at points. Defaults to None.
            create (bool, optional): whether the output file needs creating, or
                if it already exists. Defaults to True.
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
        """
        Evaluate and output field data at points.

        Args:
            field_creator (:class:`FieldCreator`): gives access to the fields.
            t (float): simulation time at which the output occurs.
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
    """Object for outputting global diagnostic data."""
    def __init__(self, filename, diagnostics, description, comm, create=True):
        """
        Args:
            filename (str): name of file to output to.
            diagnostics (:class:`Diagnostics`): the object holding and
                controlling the diagnostic evaluation.
            description (str): a description of the simulation to be included in
                the output.
            comm (:class:`MPI.Comm`): MPI communicator.
            create (bool, optional): whether the output file needs creating, or
                if it already exists. Defaults to True.
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

    def dump(self, state_fields, t):
        """
        Output the global diagnostics.

            state_fields (:class:`StateFields`): the model's field container.
            t (float): simulation time at which the output occurs.
        """

        diagnostics = []
        for fname in self.diagnostics.fields:
            field = state_fields(fname)
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


class IO(object):
    """Controls the model's input, output and diagnostics."""

    def __init__(self, domain, output, diagnostics=None, diagnostic_fields=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            output (:class:`OutputParameters`): holds and describes the options
                for outputting.
            diagnostics (:class:`Diagnostics`, optional): object holding and
                controlling the model's diagnostics. Defaults to None.
            diagnostic_fields (list, optional): an iterable of `DiagnosticField`
                objects. Defaults to None.

        Raises:
            RuntimeError: if no output is provided.
            TypeError: if `dt` cannot be cast to a :class:`Constant`.
        """
        self.domain = domain
        self.mesh = domain.mesh
        self.output = output

        if diagnostics is not None:
            self.diagnostics = diagnostics
        else:
            self.diagnostics = Diagnostics()
        if diagnostic_fields is not None:
            self.diagnostic_fields = diagnostic_fields
        else:
            self.diagnostic_fields = []

        if self.output.dumplist is None:
            self.output.dumplist = []

        self.dumpdir = None
        self.dumpfile = None
        self.to_pick_up = None

    def log_parameters(self, equation):
        """
        Logs an equation's physical parameters that take non-default values.

        Args:
            equation (:class:`PrognosticEquation`): the model's equation which
                contains any physical parameters used in the model run.
        """
        if hasattr(equation, 'parameters') and equation.parameters is not None:
            logger.info("Physical parameters that take non-default values:")
            logger.info(", ".join("%s: %s" % (k, float(v)) for (k, v) in vars(equation.parameters).items()))

    def setup_diagnostics(self, state_fields):
        """
        Prepares the I/O for computing the model's global diagnostics and
        diagnostic fields.

        Args:
            state_fields (:class:`StateFields`): the model's field container.
        """

        diagnostic_names = [diagnostic.name for diagnostic in self.diagnostic_fields]
        non_diagnostics = [fname for fname in state_fields._field_names if state_fields.field_type(fname) != "diagnostic" or fname not in diagnostic_names]

        # Set up any reference or initial fields that are necessary for diagnostics
        all_required_fields = {r for d in self.diagnostic_fields for r in d.required_fields}
        ref_fields = list(filter(lambda fname: fname[-4:] == '_bar', all_required_fields))
        init_fields = list(filter(lambda fname: fname[-5:] == '_init', all_required_fields))
        non_diagnostics = non_diagnostics + ref_fields + init_fields

        # Set up order for diagnostic fields -- filter out non-diagnostic fields
        field_deps = [(d, sorted(set(d.required_fields).difference(non_diagnostics),)) for d in self.diagnostic_fields]
        schedule = topo_sort(field_deps)
        self.diagnostic_fields = schedule

        # Set up and register all diagnostic fields
        for diagnostic in self.diagnostic_fields:
            diagnostic.setup(self.domain, state_fields)
            self.diagnostics.register(diagnostic.name)

        # Register fields for global diagnostics
        # TODO: it should be possible to specify which global diagnostics are used
        for fname in state_fields._field_names:
            if fname in state_fields.to_dump:
                self.diagnostics.register(fname)

    def setup_dump(self, state_fields, t, pick_up=False):
        """
        Sets up a series of things used for outputting.

        This prepares the model for outputting. First it checks for the
        existence the specified outputting directory, so prevent it being
        overwritten unintentionally. It then sets up the output files and the
        checkpointing file.

        Args:
            state_fields (:class:`StateFields`): the model's field container.
            t (float): the current model time.
            pick_up (bool, optional): whether to pick up the model's initial
                state from a checkpointing file. Defaults to False.

        Raises:
            IOError: if the results directory already exists, and the model is
                not picking up or running in test mode.
        """

        if any([self.output.dump_vtus, self.output.dump_nc,
                self.output.dumplist_latlon, self.output.dump_diagnostics,
                self.output.point_data, self.output.checkpoint and not pick_up]):
            # setup output directory and check that it does not already exist
            self.dumpdir = path.join("results", self.output.dirname)
            running_tests = '--running-tests' in sys.argv or "pytest" in self.output.dirname

            if self.mesh.comm.Get_rank() == 0:
                # Create results directory if it doesn't already exist
                if not path.exists(self.dumpdir):
                    makedirs(self.dumpdir)
                elif not (running_tests or pick_up):
                    # Throw an error if directory already exists, unless we
                    # are picking up or running tests
                    raise IOError(f'results directory {self.dumpdir} already exists')

            update_logfile_location(self.dumpdir, self.mesh.comm)

        if self.output.dump_vtus or self.output.dump_nc:
            # make list of fields to dump
            self.to_dump = [f for f in state_fields.fields if f.name() in state_fields.to_dump]

        # make dump counter
        self.dumpcount = itertools.count()

        if self.output.dump_vtus:
            # setup pvd output file
            outfile_pvd = path.join(self.dumpdir, "field_output.pvd")
            self.pvd_dumpfile = File(
                outfile_pvd, project_output=self.output.project_fields,
                comm=self.mesh.comm)

        if self.output.dump_nc:
            self.nc_filename = path.join(self.dumpdir, "field_output.nc")
            space_names = sorted(set([field.function_space().name for field in self.to_dump]))
            for space_name in space_names:
                self.domain.coords.register_space(self.domain, space_name)

            if pick_up:
                # Pick up t idx
                if self.mesh.comm.Get_rank() == 0:
                    nc_field_file = Dataset(self.nc_filename, 'r')
                    self.field_t_idx = len(nc_field_file['time'][:])
                    nc_field_file.close()
            else:
                # File needs creating
                self.create_nc_dump(self.nc_filename, space_names)

        # if there are fields to be dumped in latlon coordinates,
        # setup the latlon coordinate mesh and make output file
        if len(self.output.dumplist_latlon) > 0:
            mesh_ll = get_flat_latlon_mesh(self.mesh)
            outfile_ll = path.join(self.dumpdir, "field_output_latlon.pvd")
            self.dumpfile_ll = File(outfile_ll,
                                    project_output=self.output.project_fields,
                                    comm=self.mesh.comm)

            # make functions on latlon mesh, as specified by dumplist_latlon
            self.to_dump_latlon = []
            for name in self.output.dumplist_latlon:
                f = state_fields(name)
                field = Function(
                    functionspaceimpl.WithGeometry.create(
                        f.function_space(), mesh_ll),
                    val=f.topological, name=name+'_ll')
                self.to_dump_latlon.append(field)

        # we create new netcdf files to write to, unless pick_up=True and they
        # already exist, in which case we just need the filenames
        if self.output.dump_diagnostics:
            diagnostics_filename = self.dumpdir+"/diagnostics.nc"
            to_create = not (path.isfile(diagnostics_filename) and pick_up)
            self.diagnostic_output = DiagnosticsOutput(diagnostics_filename,
                                                       self.diagnostics,
                                                       self.output.dirname,
                                                       self.mesh.comm,
                                                       create=to_create)

        if len(self.output.point_data) > 0:
            # set up point data output
            pointdata_filename = self.dumpdir+"/point_data.nc"
            to_create = not (path.isfile(pointdata_filename) and pick_up)
            self.pointdata_output = PointDataOutput(pointdata_filename,
                                                    self.output.point_data,
                                                    self.output.dirname,
                                                    state_fields,
                                                    self.mesh.comm,
                                                    self.output.tolerance,
                                                    create=to_create)

            # make point data dump counter
            self.pddumpcount = itertools.count()

            # set frequency of point data output - defaults to
            # dumpfreq if not set by user
            if self.output.pddumpfreq is None:
                self.output.pddumpfreq = self.output.dumpfreq

        # if we want to checkpoint, set up the checkpointing
        if self.output.checkpoint:
            if self.output.checkpoint_method == 'dumbcheckpoint':
                # should have already picked up, so can create a new file
                self.chkpt = DumbCheckpoint(path.join(self.dumpdir, "chkpt"),
                                            mode=FILE_CREATE)
            elif self.output.checkpoint_method == 'checkpointfile':
                # should have already picked up, so can create a new file
                self.chkpt_path = path.join(self.dumpdir, "chkpt.h5")
            else:
                raise ValueError(f'checkpoint_method {self.output.checkpoint_method} not supported')

            # make list of fields to pick_up (this doesn't include
            # diagnostic fields)
            self.to_pick_up = [fname for fname in state_fields.to_pick_up]

            # make a checkpoint counter
            self.chkptcount = itertools.count()

        # dump initial fields
        if not pick_up:
            self.dump(state_fields, t)

    def pick_up_from_checkpoint(self, state_fields):
        """
        Picks up the model's variables from a checkpoint file.

        Args:
            state_fields (:class:`StateFields`): the model's field container.

        Returns:
            float: the checkpointed model time.
        """

        # -------------------------------------------------------------------- #
        # Preparation for picking up
        # -------------------------------------------------------------------- #

        # Make list of fields that must be picked up
        if self.to_pick_up is None:
            self.to_pick_up = [fname for fname in state_fields.to_pick_up]

        # Set dumpdir if has not been done already
        if self.dumpdir is None:
            self.dumpdir = path.join("results", self.output.dirname)
            update_logfile_location(self.dumpdir, self.mesh.comm)

        # Need to pick up reference profiles, but don't know which are stored
        possible_ref_profiles = []
        reference_profiles = []
        for field_name, field_type in zip(state_fields._field_names, state_fields._field_types):
            if field_type != 'reference':
                possible_ref_profiles.append(field_name)

        # -------------------------------------------------------------------- #
        # Pick up fields
        # -------------------------------------------------------------------- #

        if self.output.checkpoint:
            # Open the checkpointing file for writing
            if self.output.checkpoint_pickup_filename is not None:
                chkfile = self.output.checkpoint_pickup_filename
            elif self.output.checkpoint_method == 'dumbcheckpoint':
                chkfile = path.join(self.dumpdir, "chkpt")
            elif self.output.checkpoint_method == 'checkpointfile':
                chkfile = path.join(self.dumpdir, "chkpt.h5")

            if self.output.checkpoint_method == 'dumbcheckpoint':
                with DumbCheckpoint(chkfile, mode=FILE_READ) as chk:
                    # Recover compulsory fields from the checkpoint
                    for field_name in self.to_pick_up:
                        chk.load(state_fields(field_name), name=field_name)

                    # Read in reference profiles -- failures are allowed here
                    for field_name in possible_ref_profiles:
                        ref_name = f'{field_name}_bar'
                        ref_field = Function(state_fields(field_name).function_space(), name=ref_name)
                        try:
                            chk.load(ref_field, name=ref_name)
                            reference_profiles.append((field_name, ref_field))
                            # Field exists, so add to to_pick_up
                            self.to_pick_up.append(ref_name)
                        except RuntimeError:
                            pass

                    # Try to pick up number of initial steps for multi level scheme
                    # Not compulsory so errors allowed
                    try:
                        initial_steps = chk.read_attribute("/", "initial_steps")
                    except AttributeError:
                        initial_steps = None

                    # Finally pick up time
                    t = chk.read_attribute("/", "time")

            else:
                with CheckpointFile(chkfile, 'r') as chk:
                    mesh = self.domain.mesh
                    # Recover compulsory fields from the checkpoint
                    for field_name in self.to_pick_up:
                        field = chk.load_function(mesh, field_name)
                        state_fields(field_name).assign(field)

                    # Read in reference profiles -- failures are allowed here
                    for field_name in possible_ref_profiles:
                        ref_name = f'{field_name}_bar'
                        try:
                            ref_field = chk.load_function(mesh, ref_name)
                            reference_profiles.append((field_name, ref_field))
                            # Field exists, so add to to_pick_up
                            self.to_pick_up.append(ref_name)
                        except RuntimeError:
                            pass

                    # Try to pick up number of initial steps for multi level scheme
                    # Not compulsory so errors allowed
                    if chk.has_attr("/", "initial_steps"):
                        initial_steps = chk.get_attr("/", "initial_steps")
                    else:
                        initial_steps = None

                    # Finally pick up time
                    t = chk.get_attr("/", "time")

            # If we have picked up from a non-standard file, reset this name
            # so that we will checkpoint using normal file name from now on
            self.output.checkpoint_pickup_filename = None
        else:
            raise ValueError("Must set checkpoint True if picking up")

        return t, reference_profiles, initial_steps

    def dump(self, state_fields, t, initial_steps=None):
        """
        Dumps all of the required model output.

        This includes point data, global diagnostics and general field data to
        paraview data files. Also writes the model's prognostic variables to
        a checkpoint file if specified.

        Args:
            state_fields (:class:`StateFields`): the model's field container.
            t (float): the simulation's current time.
            initial_steps (int, optional): the number of initial time steps
                completed by a multi-level time scheme. Defaults to None.
        """
        output = self.output

        # Diagnostics:
        # Compute diagnostic fields
        for field in self.diagnostic_fields:
            field.compute()

        if output.dump_diagnostics:
            # Output diagnostic data
            self.diagnostic_output.dump(state_fields, t)

        if len(output.point_data) > 0 and (next(self.pddumpcount) % output.pddumpfreq) == 0:
            # Output pointwise data
            self.pointdata_output.dump(state_fields, t)

        # Dump all the fields to the checkpointing file (backup version)
        if output.checkpoint and (next(self.chkptcount) % output.chkptfreq) == 0:
            if self.output.checkpoint_method == 'dumbcheckpoint':
                for field_name in self.to_pick_up:
                    self.chkpt.store(state_fields(field_name), name=field_name)
                self.chkpt.write_attribute("/", "time", t)
                if initial_steps is not None:
                    self.chkpt.write_attribute("/", "initial_steps", initial_steps)
            else:
                with CheckpointFile(self.chkpt_path, 'w') as chk:
                    chk.save_mesh(self.domain.mesh)
                    for field_name in self.to_pick_up:
                        chk.save_function(state_fields(field_name), name=field_name)
                    chk.set_attr("/", "time", t)
                    if initial_steps is not None:
                        chk.set_attr("/", "initial_steps", initial_steps)

        if (next(self.dumpcount) % output.dumpfreq) == 0:
            if output.dump_nc:
                # dump fields
                self.write_nc_dump(t)

            if output.dump_vtus:
                # dump fields
                self.pvd_dumpfile.write(*self.to_dump)

                # dump fields on latlon mesh
                if len(output.dumplist_latlon) > 0:
                    self.dumpfile_ll.write(*self.to_dump_latlon)

    def create_nc_dump(self, filename, space_names):
        my_rank = self.mesh.comm.Get_rank()
        self.field_t_idx = 0

        if my_rank == 0:
            nc_field_file = Dataset(filename, 'w')
            nc_field_file.createDimension('time', None)
            nc_field_file.createVariable('time', float, ('time',))

            # Add mesh metadata
            for metadata_key, metadata_value in self.domain.metadata.items():
                # If the metadata is None or a Boolean, try converting to string
                # This is because netCDF can't take these types as options
                if type(metadata_value) in [type(None), type(True)]:
                    output_value = str(metadata_value)
                else:
                    output_value = metadata_value

                # Get the type from the metadata itself
                nc_field_file.createVariable(metadata_key, type(output_value), [])
                nc_field_file.variables[metadata_key][0] = output_value

            # Add coordinates if they are not already in the file
            for space_name in space_names:
                if space_name not in self.domain.coords.chi_coords.keys():
                    # Space not registered
                    # TODO: we should fail here, but currently there are some spaces
                    # that we can't output for so instead just skip outputting
                    pass
                else:
                    coord_fields = self.domain.coords.global_chi_coords[space_name]
                    num_points = len(self.domain.coords.global_chi_coords[space_name][0])

                    nc_field_file.createDimension('coords_'+space_name, num_points)

                    for (coord_name, coord_field) in zip(self.domain.coords.coords_name, coord_fields):
                        nc_field_file.createVariable(coord_name+'_'+space_name, float, 'coords_'+space_name)
                        nc_field_file.variables[coord_name+'_'+space_name][:] = coord_field[:]

            # Create variable for storing the field values
            for field in self.to_dump:
                field_name = field.name()
                space_name = field.function_space().name
                if space_name not in self.domain.coords.chi_coords.keys():
                    # Space not registered
                    # TODO: we should fail here, but currently there are some spaces
                    # that we can't output for so instead just skip outputting
                    logger.warning(f'netCDF outputting for space {space_name} '
                                   + 'not yet implemented, so unable to output'
                                   + '{field_name} field')
                else:
                    nc_field_file.createGroup(field_name)
                    nc_field_file[field_name].createVariable('field_values', float, ('coords_'+space_name, 'time'))

            nc_field_file.close()

    def write_nc_dump(self, t):

        comm = self.mesh.comm
        my_rank = comm.Get_rank()
        comm_size = comm.Get_size()

        # Open file to add time
        if my_rank == 0:
            nc_field_file = Dataset(self.nc_filename, 'a')
            nc_field_file['time'][self.field_t_idx] = t

        # Loop through output field data here
        num_fields = len(self.to_dump)
        for i, field in enumerate(self.to_dump):
            field_name = field.name()
            space_name = field.function_space().name

            if space_name not in self.domain.coords.chi_coords.keys():
                # Space not registered
                # TODO: we should fail here, but currently there are some spaces
                # that we can't output for so instead just skip outputting
                pass

            # -------------------------------------------------------- #
            # Scalar elements
            # -------------------------------------------------------- #
            else:
                j = 0
                # For most processors send data to first processor
                if my_rank != 0:
                    # Make a tag to uniquely identify this call
                    my_tag = comm_size*(num_fields*j + i) + my_rank
                    comm.send(field.dat.data_ro[:], dest=0, tag=my_tag)
                else:
                    # Set up array to store full data in
                    total_data_size = self.domain.coords.parallel_array_lims[space_name][comm_size-1][1]+1
                    single_proc_data = np.zeros(total_data_size)
                    # Get data for this processor first
                    (low_lim, up_lim) = self.domain.coords.parallel_array_lims[space_name][my_rank][:]
                    single_proc_data[low_lim:up_lim+1] = field.dat.data_ro[:]
                    # Receive data from other processors
                    for procid in range(1, comm_size):
                        my_tag = comm_size*(num_fields*j + i) + procid
                        incoming_data = comm.recv(source=procid, tag=my_tag)
                        (low_lim, up_lim) = self.domain.coords.parallel_array_lims[space_name][procid][:]
                        single_proc_data[low_lim:up_lim+1] = incoming_data[:]
                    # Store whole field data
                    nc_field_file[field_name].variables['field_values'][:, self.field_t_idx] = single_proc_data[:]

        if my_rank == 0:
            nc_field_file.close()

        self.field_t_idx += 1


def topo_sort(field_deps):
    """
    Perform a topological sort to determine the order to evaluate diagnostics.

    Args:
        field_deps (list): a list of tuples, pairing diagnostic fields with the
            fields that they are to be evaluated from.

    Raises:
        RuntimeError: if there is a cyclic dependency in the diagnostic fields.

    Returns:
        list: a list specifying the order in which to evaluate the diagnostics.
    """
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
