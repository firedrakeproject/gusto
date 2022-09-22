"""
Provides the model's state object, which controls IO and other core functions.

The model's :class:`State` object is defined in this module. It controls various
input/output (IO) aspects, as well as setting up the compatible finite element
spaces and holding the mesh. In some ways it acts as a bucket, holding core
parts of the model.
"""
from os import path, makedirs
import itertools
from netCDF4 import Dataset
import sys
import time
from gusto.diagnostics import Diagnostics, Perturbation, SteadyStateError
from firedrake import (FiniteElement, TensorProductElement, HDiv,
                       FunctionSpace, VectorFunctionSpace,
                       interval, Function, Mesh, functionspaceimpl,
                       File, SpatialCoordinate, sqrt, Constant, inner,
                       op2, DumbCheckpoint, FILE_CREATE, FILE_READ, interpolate,
                       CellNormal, cross, as_vector)
import numpy as np
from gusto.configuration import logger, set_log_handler
from gusto.fields import StateFields

__all__ = ["State"]


class SpaceCreator(object):
    """Object to create and hold the model's finite element spaces."""
    def __init__(self, mesh):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
        """
        self.mesh = mesh
        self.extruded_mesh = hasattr(mesh, "_base_mesh")
        self._initialised_base_spaces = False

    def __call__(self, name, family=None, degree=None, V=None):
        """
        Returns a space, and also creates it if it is not created yet.

        If a space needs creating, it may be that more arguments (such as the
        family and degree) need to be provided. Alternatively a space can be
        passed in to be stored in the creator.

        Args:
            name (str): the name of the space.
            family (str, optional): name of the finite element family to be
                created. Defaults to None.
            degree (int, optional): the degree of the finite element space to be
                created. Defaults to None.
            V (:class:`FunctionSpace`, optional): an existing space, to be
                stored in the creator object. If this is provided, it will be
                added to the creator and no other action will be taken. This
                space will be returned. Defaults to None.

        Returns:
            :class:`FunctionSpace`: the desired function space.
        """

        try:
            return getattr(self, name)
        except AttributeError:
            if V is not None:
                value = V
            elif name == "HDiv" and family in ["BDM", "RT", "CG", "RTCF"]:
                value = self.build_hdiv_space(family, degree)
            elif name == "theta":
                value = self.build_theta_space(degree)
            elif name == "DG1_equispaced":
                value = self.build_dg_space(1, variant='equispaced')
            elif family == "DG":
                value = self.build_dg_space(degree)
            elif family == "CG":
                value = self.build_cg_space(degree)
            else:
                raise ValueError(f'State has no space corresponding to {name}')
            setattr(self, name, value)
            return value

    def build_compatible_spaces(self, family, degree):
        """
        Builds the sequence of compatible finite element spaces for the mesh.

        If the mesh is not extruded, this builds and returns the spaces:
            (HDiv, DG).
        If the mesh is extruded, this builds and returns the following spaces:
            (HDiv, DG, theta).
        The 'theta' space corresponds to the vertical component of the velocity.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            degree (int): the polynomial degree of the DG space.

        Returns:
            tuple: the created compatible :class:`FunctionSpace` objects.
        """
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
        """
        Builds the :class:`FiniteElement` objects for the base mesh.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            degree (int): the polynomial degree of the DG space.
        """
        cell = self.mesh._base_mesh.ufl_cell().cellname()

        # horizontal base spaces
        self.S1 = FiniteElement(family, cell, degree+1)
        self.S2 = FiniteElement("DG", cell, degree)

        # vertical base spaces
        self.T0 = FiniteElement("CG", interval, degree+1)
        self.T1 = FiniteElement("DG", interval, degree)

        self._initialised_base_spaces = True

    def build_hdiv_space(self, family, degree):
        """
        Builds and returns the HDiv :class:`FunctionSpace`.

        Args:
            family (str): the family of the horizontal part of the HDiv space.
            degree (int): the polynomial degree of the space.

        Returns:
            :class:`FunctionSpace`: the HDiv space.
        """
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
        return FunctionSpace(self.mesh, V_elt, name='HDiv')

    def build_dg_space(self, degree, variant=None):
        """
        Builds and returns the DG :class:`FunctionSpace`.

        Args:
            degree (int): the polynomial degree of the space.
            variant (str): the variant of the underlying :class:`FiniteElement`
                to use. Defaults to None, which will call the default variant.

        Returns:
            :class:`FunctionSpace`: the DG space.
        """
        if self.extruded_mesh:
            if not self._initialised_base_spaces or self.T1.degree() != degree or self.T1.variant() != variant:
                cell = self.mesh._base_mesh.ufl_cell().cellname()
                S2 = FiniteElement("DG", cell, degree, variant=variant)
                T1 = FiniteElement("DG", interval, degree, variant=variant)
            else:
                S2 = self.S2
                T1 = self.T1
            V_elt = TensorProductElement(S2, T1)
        else:
            cell = self.mesh.ufl_cell().cellname()
            V_elt = FiniteElement("DG", cell, degree, variant=variant)
        name = f'DG{degree}_equispaced' if variant == 'equispaced' else f'DG{degree}'
        return FunctionSpace(self.mesh, V_elt, name=name)

    def build_theta_space(self, degree):
        """
        Builds and returns the 'theta' space.

        This corresponds to the non-Piola mapped space of the vertical component
        of the velocity. The space will be discontinuous in the horizontal but
        continuous in the vertical.

        Args:
            degree (int): degree of the corresponding density space.

        Raises:
            AssertionError: the mesh is not extruded.

        Returns:
            :class:`FunctionSpace`: the 'theta' space.
        """
        assert self.extruded_mesh
        if not self._initialised_base_spaces:
            cell = self.mesh._base_mesh.ufl_cell().cellname()
            self.S2 = FiniteElement("DG", cell, degree)
            self.T0 = FiniteElement("CG", interval, degree+1)
        V_elt = TensorProductElement(self.S2, self.T0)
        return FunctionSpace(self.mesh, V_elt, name='Vtheta')

    def build_cg_space(self, degree):
        """
        Builds the continuous scalar space at the top of the de Rham complex.

        Args:
            degree (int): degree of the continuous space.

        Returns:
            :class:`FunctionSpace`: the continuous space.
        """
        return FunctionSpace(self.mesh, "CG", degree, name=f'CG{degree}')


class PointDataOutput(object):
    """Object for outputting field point data."""
    def __init__(self, filename, ndt, field_points, description,
                 field_creator, comm, tolerance=None, create=True):
        """
        Args:
            filename (str): name of file to output to.
            ndt (int): number of time points to output at. TODO: remove as this
                is unused.
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

    def dump(self, state, t):
        """
        Output the global diagnostics.

            state (:class:`State`): the model's state object.
            t (float): simulation time at which the output occurs.
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
    """Keeps the model's mesh and variables, and controls its IO."""

    def __init__(self, mesh, dt,
                 output=None,
                 parameters=None,
                 diagnostics=None,
                 diagnostic_fields=None):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
            dt (:class:`Constant`): the time taken to perform a single model
                step. If a float or int is passed, it will be cast to a
                :class:`Constant`.
            output (:class:`OutputParameters`, optional): holds and describes
                the options for outputting. Defaults to None.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters. Defaults to None.
            diagnostics (:class:`Diagnostics`, optional): object holding and
                controlling the model's diagnostics. Defaults to None.
            diagnostic_fields (list, optional): an iterable of `DiagnosticField`
                objects. Defaults to None.

        Raises:
            RuntimeError: if no output is provided.
            TypeError: if `dt` cannot be cast to a :class:`Constant`.
        """

        if output is None:
            # TODO: maybe this shouldn't be an optional argument then?
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

        self.dumpdir = None
        self.dumpfile = None
        self.to_pickup = None

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

        # setup logger
        logger.setLevel(output.log_level)
        set_log_handler(mesh.comm)
        if parameters is not None:
            logger.info("Physical parameters that take non-default values:")
            logger.info(", ".join("%s: %s" % (k, float(v)) for (k, v) in vars(parameters).items()))

        #  Constant to hold current time
        self.t = Constant(0.0)
        if type(dt) is Constant:
            self.dt = dt
        elif type(dt) in (float, int):
            self.dt = Constant(dt)
        else:
            raise TypeError(f'dt must be a Constant, float or int, not {type(dt)}')

    def setup_diagnostics(self):
        """Concatenates the various types of diagnostic field."""
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
        Sets up a series of things used for outputting.

        This prepares the model for outputting. First it checks for the
        existence the specified outputting directory, so prevent it being
        overwritten unintentionally. It then sets up the output files and the
        checkpointing file.

        Args:
            t (float): the current model time.
            tmax (float): the end time of the model's simulation.
            pickup (bool, optional): whether to pick up the model's initial
                state from a checkpointing file. Defaults to False.

        Raises:
            IOError: if the results directory already exists, and the model is
                not picking up or running in test mode.
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
                    functionspaceimpl.WithGeometry.create(
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
            ndt = int(tmax/float(self.dt))
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
        # checkpoint file, setup the checkpointing
        if self.output.checkpoint:
            if not pickup:
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
        """Picks up the model's variables from a checkpoint file."""
        # TODO: this duplicates some code from setup_dump. Can this be avoided?
        # It is because we don't know if we are picking up or setting dump first
        if self.to_pickup is None:
            self.to_pickup = [f for f in self.fields if f.name() in self.fields.to_pickup]
        # Set dumpdir if has not been done already
        if self.dumpdir is None:
            self.dumpdir = path.join("results", self.output.dirname)

        if self.output.checkpoint:
            # Open the checkpointing file for writing
            if self.output.checkpoint_pickup_filename is not None:
                chkfile = self.output.checkpoint_pickup_filename
            else:
                chkfile = path.join(self.dumpdir, "chkpt")
            with DumbCheckpoint(chkfile, mode=FILE_READ) as chk:
                # Recover all the fields from the checkpoint
                for field in self.to_pickup:
                    chk.load(field)
                t = chk.read_attribute("/", "time")
            # Setup new checkpoint
            self.chkpt = DumbCheckpoint(path.join(self.dumpdir, "chkpt"), mode=FILE_CREATE)
        else:
            raise ValueError("Must set checkpoint True if pickup")

        return t

    def dump(self, t):
        """
        Dumps all of the required model output.

        This includes point data, global diagnostics and general field data to
        paraview data files. Also writes the model's prognostic variables to
        a checkpoint file if specified.

        Args:
            t (float): the simulation's current time.
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
        Initialise the state's prognostic variables.

        Args:
            initial_conditions (list): an iterable of pairs: (field_name, expr),
                where 'field_name' is the string giving the name of the
                prognostic field and expr is the :class:`ufl.Expr` whose value
                is used to set the initial field.
        """
        for name, ic in initial_conditions:
            f_init = getattr(self.fields, name)
            f_init.assign(ic)
            f_init.rename(name)

    def set_reference_profiles(self, reference_profiles):
        """
        Initialise the state's reference profiles.

        reference_profiles (list): an iterable of pairs: (field_name, expr),
                where 'field_name' is the string giving the name of the
                reference profile field expr is the :class:`ufl.Expr` whose
                value is used to set the reference field.
        """
        for name, profile in reference_profiles:
            if name+'bar' in self.fields:
                # For reference profiles already added to state, allow
                # interpolation from expressions
                ref = self.fields(name+'bar')
            elif isinstance(profile, Function):
                # Need to add reference profile to state so profile must be
                # a Function
                ref = self.fields(name+'bar', space=profile.function_space(), dump=False)
            else:
                raise ValueError(f'When initialising reference profile {name}'
                                 + ' the passed profile must be a Function')
            ref.interpolate(profile)


def get_latlon_mesh(mesh):
    """
    Construct a planar latitude-longitude mesh from a spherical mesh.

    Args:
        mesh (:class:`State`): the mesh on which the simulation is performed.
    """
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
