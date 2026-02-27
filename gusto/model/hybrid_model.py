from firedrake import (Function, as_vector, CheckpointFile, assemble, dx,
                       VectorFunctionSpace, FunctionSpace, VertexOnlyMesh,
                       interpolate, Constant)
from firedrake.adjoint import (Control, ReducedFunctional, set_working_tape,
                               continue_annotation)
from firedrake.ml.pytorch import to_torch, from_torch, fem_operator

from torch.nn import Module, Sequential, Linear, ReLU
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from tqdm.auto import tqdm, trange

from collections import namedtuple, defaultdict
import numpy as np
import os
import torch

continue_annotation()

class HybridModel(object):

    def __init__(self, pde_model, ml_model, input_fields, fields_to_adjust,
                 data_dir):
        """
        Args:
            pde_model: PDE model (Gusto model class?)
            ml_model: PyTorch model
            input_fields, (list):
            fields_to_adjust (list): list of names of fields to adjust
                with ml model
            data_dir (str): directory where to save / load data to / from
        """

        self.pde_model = pde_model
        self.ml_model = ml_model
        self.input_fields = input_fields
        self.fields_to_adjust = fields_to_adjust
        self.data_dir = data_dir

        self.batch_size = 1

        # Set double precision in ml_model to match types
        ml_model.double()
        # Move ml_model to device
        self.device = "cpu"
        ml_model.to(self.device)

        fs = pde_model.equation.X.function_space()
        self.pde_out = Function(fs)
        self.ml_out = Function(fs)
        self.x_predicted = Function(fs)
        self.x_target = Function(fs)

    def assemble_cost_function(self, x_predicted, x_target):
        """
        ML cost function
        """
        return assemble(0.5 * (x_predicted - x_target)**2 * dx)

    def assemble_prediction(self, dyn, ml):
        """
        This adds the ml prediction to the dynamics predictions
        """

        ml_idx = 0
        for field_name in self.fields_to_adjust:
            idx = self.pde_model.equation.field_names.index(field_name)
            if field_name == "u":
                self.ml_out.subfunctions[idx].project(
                    as_vector([ml[ml_idx], ml[ml_idx+1]])
                )
                ml_idx += 2
            else:
                self.ml_out.subfunctions[idx].interpolate(ml[ml_idx])
                ml_idx += 1

        self.x_predicted.interpolate(dyn + self.ml_out)

        return self.x_predicted

    def generate_data(self, initial_conditions, ndt):
        """
        Generate data using the PDE model.
        Args:
            initial_conditions, (list): a list of tuples specifying initial
                conditions. Each tuple should contain tuples specifying the
                pairs (field_name, initial_condition) where field_name is a
                string corresponding to a prognostic field in the PDE model
                and initial_condition is a ufl.Expr or Firedrake function that
                specifies the initial values of that field.
            ndt, (int): number of timesteps to run the model for
        """

        self.ndt = ndt

        # for each initial condition, we make a copy of the PDE model
        # class, initialise it, timestep it and save the data in a
        # checkpoint file.
        # CHECK: is it ok to reuse the model class with different ics?
        pde_model = self.pde_model
        # CHECK: there must be a better way to change the model output settings?
        output = pde_model.stepper.io.output
        # TODO: use self.input_fields to prescribe which fields to dump
        output.checkpoint = True
        # TODO: probably should be input arg, along with e.g. spinup_steps
        output.chkptfreq = 1
        output.multichkpt = True
        # create list to store output directories for data processing
        self.dir_list = []
        for i, ic in enumerate(initial_conditions):
            output.dirname = os.path.join(self.data_dir, f"test_train_{i}")
            self.dir_list.append(output.dirname)
            stepper = pde_model.stepper
            for field_name, field_ic in ic:
                field = stepper.fields(field_name)
                if field_name == "u":
                    field.project(field_ic)
                else:
                    field.interpolate(field_ic)
            stepper.run(t=0, tmax=ndt*float(pde_model.domain.dt))

    def process_data(self, ndt=None, dir_list=None):
        """
        Processes multiple global checkpoint data files into point data
        and global data in the format required for training the model
        Args:
           ndt, (int):
           dir_list, (list):
        """
        if ndt is None:
            ndt = self.ndt
        if dir_list is None:
            dir_list = self.dir_list
        point_data = []
        global_data = []
        point_times = []
        point_labels = []
        global_times = []
        global_labels = []
        simulation_numbers = []
        for i, dirname in enumerate(dir_list):
            chkpt_file = os.path.join(dirname, "chkpt.h5")
            with CheckpointFile(chkpt_file, 'r') as chkfile:
                mesh = chkfile.load_mesh(name='mesh')
                for n in range(ndt):
                    t = chkfile.get_timestepping_history(mesh, 'u').get('time')[n]
                    fields = []
                    for field_name in self.input_fields:
                        if field_name == "u":
                            u = chkfile.load_function(mesh, field_name, n)
                            # Extract components of u
                            # TODO: generalise Vdg function space
                            Vdg = FunctionSpace(mesh, "DG", 1)
                            # TODO: this only works on the plane
                            u0 = Function(Vdg, name="u").interpolate(u[0])
                            u1 = Function(Vdg, name="v").interpolate(u[1])
                            fields.append(u0)
                            fields.append(u1)
                        else:
                            fields.append(chkfile.load_function(mesh, field_name, n))
                    point_values, coords = self.point_evaluation(mesh, fields)
                    for data_tuple in zip(*point_values):
                        point_data.append(data_tuple)
                        # Save labels to identify the time and simulation
                        # for the data
                        point_times.append(t)
                        # TODO label is maybe not necessary - the same
                        # information can be accessed from a
                        # combination of the time and simulation
                        # number
                        point_labels.append(n)

                    # Append fields to the global data list
                    global_data.append(fields)
                    # Save labels to identify the time and simulation
                    # for the data
                    global_times.append(t)
                    # TODO label is maybe not necessary - the same
                    # information can be accessed from a
                    # combination of the time and simulation
                    # number
                    global_labels.append(n)
                    simulation_numbers.append(i)

        # split into training and testing data
        global_train, global_test, point_train, point_test = \
            self.train_test_split(global_data, point_data, 0.8)

        # write global training data to checkpoint file
        filename = os.path.join(self.data_dir, "global_train_data.h5")
        n_global_train = len(global_train)
        with CheckpointFile(filename, "w") as afile:
            afile.h5pyfile["n"] = len(global_train)
            afile.save_mesh(mesh)
            for i, fields in enumerate(global_train):
                for field in fields:
                    afile.save_function(field, idx=i, name=field.name())
            afile.set_attr("/", "times", global_times[:n_global_train])
            afile.set_attr("/", "sims", simulation_numbers[:n_global_train])
            afile.set_attr("/", "labels", global_labels[:n_global_train])
            afile.set_attr("/", "locations", coords)

        # write global testing data to checkpoint file
        filename = os.path.join(self.data_dir, "global_test_data.h5")
        with CheckpointFile(filename, "w") as afile:
            afile.h5pyfile["n"] = len(global_test)
            afile.save_mesh(mesh)
            for i, fields in enumerate(global_test):
                for field in fields:
                    afile.save_function(field, idx=i, name=field.name())
            afile.set_attr("/", "times", global_times[n_global_train:])
            afile.set_attr("/", "sims", simulation_numbers[n_global_train:])
            afile.set_attr("/", "labels", global_labels[n_global_train:])
            afile.set_attr("/", "locations", coords)

        filename = os.path.join(self.data_dir, "point_train_data")
        n_point_train = len(point_train)
        point_times_train = np.array([point_times[:n_point_train]])
        point_labels_train = np.array([point_labels[:n_point_train]])
        np.save(filename,
                np.concatenate([point_train,
                                point_times_train.T,
                                point_labels_train.T], axis=1))
        filename = os.path.join(self.data_dir, "point_test_data")
        point_times_test = np.array([point_times[n_point_train:]])
        point_labels_test = np.array([point_labels[n_point_train:]])
        np.save(filename,
                np.concatenate([point_test,
                                point_times_test.T,
                                point_labels_test.T], axis=1))

    def point_evaluation(self, mesh, fields):
        """
        Evaluates fields at coordinates of DG space.
        Args:
            mesh: mesh from checkpoint file
            fields: list of Firedrake Functions to evaluate at points
        """
        # find point locations
        # TODO: generalise Vdg function space
        Vdg = FunctionSpace(mesh, "DG", 1)
        W = VectorFunctionSpace(mesh, Vdg.ufl_element())
        X = assemble(interpolate(mesh.coordinates, W))
        coords = X.dat.data_ro

        # create VOM
        vom = VertexOnlyMesh(mesh, coords)
        P0DG = FunctionSpace(vom, "DG", 0)

        point_values = []
        for field in fields:
            field_at_points = assemble(interpolate(field, P0DG))
            point_values.append(field_at_points.dat.data_ro)

        # append coordinates
        # TODO only works on plane
        point_values.append(coords[:, 0])
        point_values.append(coords[:, 1])

        return point_values, coords

    def train_test_split(self, global_data, point_data, train_proportion):
        """
        Args:
            global_data (list): a list of all global data
            point_data (list): a list of all point data
            train_proportion (float): the proportion (between 0 and 1) of the
                data to use as the training set. The remaining data become the
                test set.
        """
        total_global_samples = len(global_data)
        total_point_samples = len(point_data)
        pp_sample = int(total_point_samples/total_global_samples)

        n_global_train = int(train_proportion * total_global_samples)
        n_point_train = int(n_global_train * pp_sample)

        global_train = global_data[:n_global_train]
        global_test = global_data[n_global_train:]

        point_train = point_data[:n_point_train]
        point_test = point_data[n_point_train:]

        return global_train, global_test, point_train, point_test

    def advance_pde(self, xin, xout, ndt):

        for field_name in self.pde_model.equation.field_names:
            idx = self.pde_model.equation.field_names.index(field_name)
            self.pde_model.stepper.fields(field_name).assign(xin.subfunctions[idx])
        tmax = float(Constant(self.stepper.domain.dt) * ndt)

        self.stepper.run(0, tmax)

        for field_name in self.pde_model.equation.field_names:
            idx = self.pde_model.equation.field_names.index(field_name)
            xin.subfunctions[idx].assign(self.pde_model.stepper.fields(field_name))

    def pde_to_ml(self):
        """
        Converts hybrid prediction into tensors for next ml model step
        """
        tensors = []
        for field in self.input_fields:
            tensors.append()

    def subsample_point_data(self, point_train_dataloader):
        """
        Args:
            point_train_dataloader (:class: `Dataloader`): a Dataloader object
            with all the point data, to be separated by global sample where
            one global sample corresponds to the solution at one a single
            time of a simulation.
        """
        labels = []
        for step_num, point_sample in enumerate(point_train_dataloader):
            point_label = point_sample[:, -1].item()
            labels.append(point_label)
            # define a list of lists of indices where all the labels match
            index_list = []
            for lables, indices in sorted(self.list_duplicates(labels)):
                index_list.append(indices)

            # use index_list to sub-sample the point data by labels
            subsets = []
            for l in index_list:
                subset = Subset(point_train_dataloader.dataset, l)
                subsets.append(subset)

        # return a list of datasets, where all examples in each
        # dataset belong to one global sample
        return subsets

    def list_duplicates(self, labels):
        # This method returns a list of tuples of (label, [indices])
        # where [indices] is a list of where the label occurs
        """
        Args:
        labels (list): a list of integer simulation labels
        """
        tally = defaultdict(list)
        for i, item in enumerate(labels):
            tally[item].append(i)
        return (indices for indices in tally.items() if len(indices) > 1)

    def forward_pass(self):
        """
        Forward pass of the ml model. Returns a list of tensors.
        """

    def train(self, point_training_data, global_training_data,
              point_test_data, global_test_data,
              learning_rate=5e-5, epochs=20, rollout_length=2):
        """
        Train the hybrid model.
        """
        # Create dataset classes and dataloader classes for reading in
        # training and testing data
        point_train_dataset = PointDataset(
            numpy_data=point_training_data,
            data_dir=self.data_dir
        )
        point_train_dl = DataLoader(
            point_train_dataset, batch_size=self.batch_size, shuffle=False
        )
        global_train_dataset = GustoGlobalDataset(
            dataset=global_training_data,
            data_dir=self.data_dir,
            field_names=self.input_fields,
            attr_names=""
        )
        global_train_dl = DataLoader(
            global_train_dataset, batch_size=self.batch_size,
            collate_fn=global_train_dataset.collate, shuffle=False
        )

        point_test_dataset = PointDataset(
            numpy_data=point_test_data,
            data_dir=self.data_dir
        )
        point_test_dl = DataLoader(
            point_test_dataset, batch_size=self.batch_size, shuffle=False
        )
        global_test_dataset = GustoGlobalDataset(
            dataset=global_test_data,
            data_dir=self.data_dir,
            field_names=self.input_fields,
            attr_names=""
        )
        global_test_dl = DataLoader(
            global_test_dataset, batch_size=self.batch_size,
            collate_fn=global_test_dataset.collate, shuffle=False
        )

        #
        with set_working_tape() as tape:
            F_pred = ReducedFunctional(
                self.assemble_prediction(self.pde_out, self.ml_out),
                [Control(self.pde_out), Control(self.ml_out)],
                eval_cb_pre=self.advance_pde
            )
            G = fem_operator(F_pred)

            #F_ml_in = ReducedFunctional(self.pde_to_ml(self.x_predicted),
            #                            [Control(self.x_predicted)])
            #J = fem_operator(F_ml_in)

            F_loss = ReducedFunctional(
                self.assemble_cost_function(self.x_predicted, self.x_target),
                [Control(self.x_predicted), Control(self.x_target)])
            H = fem_operator(F_loss)

        optimiser = optim.AdamW(self.ml_model.parameters(),
                                lr=learning_rate, eps=1e-8)

        best_error = np.finfo(float).max
        max_grad_norm = 1.

        # Training loop
        for epoch_num in trange(epochs):
            self.ml_model.train()
            total_loss = 0.0
            train_steps = len(global_train_dl)
            point_train_data_subsets = self.subsample_point_data(point_train_dl)

            if len(point_train_data_subsets) != train_steps:
                print("The number of data subsets does not match the number of global samples")

            for step_num, (subset, global_sample) in tqdm(enumerate(list(zip(point_train_data_subsets, global_train_dl))),
                                                          total=train_steps):

                self.ml_model.zero_grad()

                batch = BatchedElement(*[x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in global_sample])
                tensors = []
                for field in self.input_fields:
                    tensors.append(batch.__getattr__(field))

                # Set initial values for dynamics and network
                nn_in = subset
                dyn_in = torch.cat(tensors, dim=1)

                # The index of the sample
                sample_idx = batch.idx[0]

                for rollout_step in range(rollout_length):
                    # Produce a network prediction for u and D using the same
                    # initial condition that the dynamics will see
                    nn_out = self.forward_pass(nn_in, self.ml_model,
                                               batch_size,
                                               rollout_step)

                    # Run forward PDE model for ndt timesteps and add
                    # the network forcings
                    pred_tensor = G(dyn_in, nn_out)

                    # Prepare input for next network call
                    next_tensor = J(pred_tensor)

                    # Stack tensors
                    nn_in = torch.stack(*next_tensor, dim=2)

                    # Create input for the next dynamics call
                    dyn_in = pred_tensor

                # The target is the u solution the rollout length time later
                target_idx = sample_idx + rollout_length

                # Access the sample using its index, while the target
                # index is less than the length of indices
                try:
                    target_sample = list(global_train_dl)[target_idx]
                except:
                    pass
                target_batch = BatchedElement(*[x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in target_sample])

                # Check that the target comes from the same simulation
                # (the last two will never)
                IC_sim_no = target_batch.s[0][sample_idx]
                try:
                    target_sim_no = target_batch.s[0][target_idx]
                except:
                    pass

                # where these are not the same we don't use the data
                # if they are the same define a target, set up a loss
                # function and backprop
                if IC_sim_no == target_sim_no:
                    targets = []
                    for field in self.input_fields:
                        targets.append(target_batch.getattr(field))
                    target_tensor = torch.cat(targets, dim=1)

                    # Define L2-loss using Firedrake
                    loss = H(pred_tensor, target_tensor)
                    total_loss += loss.item()

                    # Backprop and perform Adam optimisation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.ml_model.parameters(),
                        max_norm=max_grad_norm)
                    optimiser.step()

            # Evaluate this version of the model on the test set
            error = self.evaluate(self.ml_model, self.device,
                                  point_test_dl, global_test_dl,
                                  pde_model=self.pde_model,
                                  disable_tqdm=False, write_out_results=False,
                                  rollout_length=rollout_length, dt=dt, ndt=ndt)

            # Save best-performing model
            if error < best_error:
                best_error = error

                # Save model
                # Take care of distributed/parallel training
                model_to_save = (self.ml_model.module if hasattr(self.ml_model, "module") else self.ml_model)
                checkpoint = {
                    'epoch': epoch_num + 1,
                    'state_dict': model_to_save.state_dict(),
                    'optimiser': optimiser.state_dict()
                }
                model_name = f"epoch-{epoch_num}-error_{best_error:.5f}.pt"
                torch.save(checkpoint, os.path.join(self.data_dir, model_name))

    def evalute(self, filename=None):
        """
        Evaluate the hybrid model
        Args:
            filename (str, optional): where to find validation data. Defaults
            to None in which case use validation data generated by
            generate_data method
        """
        self.ml_model.load_state_dict(trained_ml_model["state_dict"])

class GustoGlobalDataset(Dataset):

    def __init__(self, dataset, data_dir, field_names, attr_names):
        """
        Args:
            dataset (string): the name of the .h5 file with the data
            data_dir (string): the path to where the data are saved
        """
        # Check dataset directory
        data_file = os.path.join(data_dir, dataset)
        if not os.path.exists(data_file):
            raise ValueError(f"Dataset directory {os.path.abspath(data_file)} does not exist")

        # Get mesh and batch elements (Firedrake functions)
        self.attr_items = attr_names
        data = []
        with CheckpointFile(data_file, "r") as afile:
            n = int(np.array(afile.h5pyfile["n"]))
            # Load mesh
            mesh = afile.load_mesh("mesh")
            # Load data
            fields = []
            attrs = []
            for i in range(n):
                for field_name in field_names:
                    fields.append(afile.load_function(
                        mesh, field_name, idx=i))
                for attr_name in attr_names:
                    attrs.append(afile.get_attr("/", key=attr_name))

                data.append(tuple(fields + attrs))

        self.mesh = mesh

        self.field_names = field_names
        self.batch_elements_fd = data
        self.fs = self.batch_elements_fd[0][0].function_space()

    def __len__(self):
        return len(self.batch_elements_fd)

    def __getitem__(self, idx):
        """
        Args:
            idx (float): the index of the sample
        """
        batch_element = self.batch_elements_fd[idx]
        name_list = []
        values = []
        # Convert Firedrake functions to PyTorch tensors
        for be in batch_element:
            if isinstance(be, Function):
                name_list.append(be.name)
                values.append(to_torch(be))
                name_list.append(be.name+"_fd")
                values.append(be)
            else:
                name_list.append(be.key)
                values.append(be)

        name_list.append("idx")
        values.append(idx)

        BatchElement = namedtuple("BatchElement", name_list)
        BatchElement(*values)

        return BatchElement

    def collate(self, batch_elements):

        batch_size = len(batch_elements)
        tensors = {}
        for field_name in self.field_names:
            be_field = batch_elements[0].__getattr__(field_name)
            n = be.size(-1)
            tensors[field_name] = torch.zeros(batch_size, n, dtype= be.dtype)

        name_list = []
        for be in batch_elements[0]:
            if isinstance(be, Function):
                name_list.append(field_name)
                name_list.append(be.name+"_fd")
            else:
                name_list.append(be.key)

        values = []
        for i, be in enumerate(batch_elements):
            for field_name in self.field_names:
                tensors[field_name][i, :] = be.__getattr__(field_name)
                field = be.__getattr__(field_name)
            attrs = []
            for attr in self.attr_names:
                attrs.append(be.__getattr__(attr))
            attrs.append(be.idx)
            values.append(tensors[field_name], field, attrs)

        BatchedElement = namedtuple("BatchedElement", name_list)
        BatchedElement(*values)

        return BatchedElement


class PointDataset(Dataset):
    """
    Dataset reader for data generated point-wise from a PDE
    solution. The pointwise data should be saved as numpy arrays.
    """

    def __init__(self, numpy_data, data_dir):
        """
        Args:
            numpy_data (string): the name of the .npy file with the data
            data_dir (string, optional): the path to where the numpy data are saved
        """
        # Check dataset directory
        datafile = os.path.join(data_dir, numpy_data)
        self.numpy_list = np.load(datafile)

    def __len__(self):
        return len(self.numpy_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (float): the index of the sample
        """
        # Make the sample a numpy array first
        numpy_sample = np.array(self.numpy_list[idx])
        # Convert the numpy array to a PyTorch tensor
        tensor_sample = torch.from_numpy(numpy_sample)
        return tensor_sample


class PointNN(Module):
    """A simple nn-based model to deal with general point-based problems"""

    def __init__(self, n_in, n_out):
        """
        Args:
            n_in (int): number of input parameters
            n_out (int): number of output parameters
        """

        super().__init__()

        self.nn_encoder = Sequential(Linear(n_in, 32),
                                     ReLU(True),
                                     Linear(32, 64),
                                     ReLU(True),
                                     Linear(64, 128),
                                     ReLU(True))

        self.nn_decoder = Sequential(Linear(128, 64),
                                     ReLU(True),
                                     Linear(64, 32),
                                     ReLU(True),
                                     Linear(32, n_out))

    def forward(self, input_tensor):

        # CNN encoder-decoder
        z = self.nn_encoder(input_tensor)
        y = self.nn_decoder(z)

        return y
