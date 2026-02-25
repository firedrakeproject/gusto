from firedrake import (Function, as_vector, CheckpointFile, assemble, dx,
                       VectorFunctionSpace, FunctionSpace, VertexOnlyMesh,
                       interpolate)
from firedrake.adjoint import Control, ReducedFunctional
from firedrake.ml.pytorch import to_torch, from_torch, fem_operator
from torch.nn import Module, Sequential, Linear, ReLU
from torch.utils.data import DataLoader, Dataset

from collections import namedtuple
import numpy as np
import os
import torch


class HybridModel(object):

    def __init__(self, pde_model, ml_model, fields_to_adjust,
                 data_dir):
        """
        Args:
            pde_model: PDE model (Gusto model class?)
            ml_model: PyTorch model
            fields_to_adjust (list): list of names of fields to adjust
                with ml model
            data_dir (str): directory where to save / load data to / from
        """

        self.pde_model = pde_model
        self.ml_model = ml_model
        self.fields_to_adjust = fields_to_adjust
        self.data_dir = data_dir

        self.batch_size = 1

        # Set double precision in ml_model to match types
        ml_model.double()
        # Move ml_model to device
        device = "cpu"
        ml_model.to(device)

        fs = pde_model.equation.X.function_space()
        self.ml_out = Function(fs)
        self.x_predicted = Function(fs)

    def assemble_cost_function(self, x_predicted, x_target):
        """
        ML cost function
        """
        return assemble(0.5 * (x_predicted, x_target)**2 * dx)

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

        self.prediction.interpolate(dyn + self.ml_out)

    def generate_data(self, initial_conditions, data_fields, ndt):
        """
        Generate data using the PDE model.
        Args:
            initial_conditions, (list): a list of tuples specifying initial
                conditions. Each tuple should contain tuples specifying the
                pairs (field_name, initial_condition) where field_name is a
                string corresponding to a prognostic field in the PDE model
                and initial_condition is a ufl.Expr or Firedrake function that
                specifies the initial values of that field.
            data_fields, (list): a list of strings specifying the fields we
                need to store and process data for.
            ndt, (int): number of timesteps to run the model for
        """

        # for each initial condition, we make a copy of the PDE model
        # class, initialise it, timestep it and save the data in a
        # checkpoint file.
        # CHECK: is it ok to reuse the model class with different ics?
        pde_model = self.pde_model
        # CHECK: there must be a better way to change the model output settings?
        output = pde_model.stepper.io.output
        output.checkpoint = True
        # TODO: probably should be input arg, along with e.g. spinup_steps
        output.chkptfreq = 1
        output.multichkpt = True
        # create list to store output directories for data processing
        dir_list = []
        for i, ic in enumerate(initial_conditions):
            output.dirname = os.path.join(self.data_dir, f"test_train_{i}")
            dir_list.append(output.dirname)
            stepper = pde_model.stepper
            for field_name, field_ic in ic:
                field = stepper.fields(field_name)
                if field_name == "u":
                    field.project(field_ic)
                else:
                    field.interpolate(field_ic)
            stepper.run(t=0, tmax=ndt*float(pde_model.domain.dt))

        point_data = []
        global_data = []
        for i, dirname in enumerate(dir_list):
            chkpt_file = os.path.join("results/", self.data_dir,
                                      f"test_train_{i}", "chkpt.h5")
            with CheckpointFile(chkpt_file, 'r') as chkfile:
                mesh = chkfile.load_mesh(name='mesh')
                fields = []
                for n in range(ndt):
                    t = chkfile.get_timestepping_history(mesh, 'u').get('time')[n]
                    for field_name in data_fields:
                        if field_name == "u":
                            u = chkfile.load_function(mesh, field_name, n)
                            # Extract components of u
                            # TODO: generalise Vdg function space
                            Vdg = FunctionSpace(mesh, "DG", 1)
                            # TODO: this only works on the plane
                            u0 = Function(Vdg).interpolate(u[0])
                            u1 = Function(Vdg).interpolate(u[1])
                            fields.append(u0)
                            fields.append(u1)
                        else:
                            fields.append(chkfile.load_function(mesh, field_name, n))
                    point_values = self.point_evaluation(mesh, fields)
                    for data_tuple in zip(*point_values):
                        point_data.append((*data_tuple, t, n))

                    # Append fields to the global data list
                    global_data.append((*fields, t, i, n))

    def point_evaluation(self, mesh, fields):

        # find point locations
        L2 = self.pde_model.domain.spaces("L2")
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

        return point_values

    def forward_pass(self):
        """
        Forward pass of the ml model. Returns a list of tensors.
        """

    def train(self, data_dir, point_training_data, global_training_data,
              point_test_data, global_test_data,
              learning_rate=5e-5, epochs=20, rollout_length=2):
        """
        Train the hybrid model.
        """
        # Create dataset classes and dataloader classes for reading in
        # training and testing data
        point_train_dataset = PointDataset(
            numpy_data=os.path.join(data_dir, point_training_data),
            data_dir=data_dir
        )
        point_train_dl = DataLoader(
            point_train_dataset, batch_size=self.batch_size, shuffle=False
        )
        global_train_dataset = GustoGlobalDataset(
            dataset=os.path.join(data_dir, global_training_data),
            data_dir=data_dir
        )
        global_train_dl = DataLoader(
            global_train_dataset, batch_size=self.batch_size,
            collate_fn=global_train_dataset.collate, shuffle=False
        )

        point_test_dataset = PointDataset(
            numpy_data=os.path.join(data_dir, point_test_data),
            data_dir=data_dir
        )
        point_test_dl = DataLoader(
            point_test_dataset, batch_size=self.batch_size, shuffle=False
        )
        global_test_dataset = GustoGlobalDataset(
            dataset=os.path.join(data_dir, global_test_data),
            data_dir=data_dir
        )
        global_test_dl = DataLoader(
            global_test_dataset, batch_size=self.batch_size,
            collate_fn=self.global_test_dataset.collate, shuffle=False
        )

        # 
        with set_working_tape() as tape:
            F_pred = ReducedFunctional(
                self.assemble_prediction(dyn_out, nn_u0, nn_u1, nn_D),
                [Control(dyn_out), Control(nn_u0), Control(nn_u1), Control(nn_D)],
                eval_cb_pre=advance_pde)
            G = fem_operator(F_pred)
            F_u0_in = ReducedFunctional(pred_to_u0_input(pred_func),
                                        [Control(pred_func)])
            J_u0 = fem_operator(F_u0_in)
            F_loss = ReducedFunctional(assemble_L2_error(f_pred, f_exact),
                                       [Control(f_pred), Control(f_exact)])
            H = fem_operator(F_loss)

        optimiser = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    def evalute(self, filename=None):
        """
        Evaluate the hybrid model
        Args:
            filename (str, optional): where to find validation data. Defaults
            to None in which case use validation data generated by
            generate_data method
        """


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
        return self.batch_elements_fd

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

        BatchedElement
        return BatchedElement


class PointDataset(Dataset):
    """
    Dataset reader for data generated point-wise from a PDE
    solution. The pointwise data should be saved as numpy arrays.
    """

    def __init__(self, numpy_data, data_dir=None):
        """
        Args:
            numpy_data (string): the name of the .npy file with the data
            data_dir (string, optional): the path to where the numpy data are saved
        """
        # Check dataset directory
        if data_dir is not None:
            dataset_dir = os.path.join(data_dir, "datasets", numpy_data)
            if not os.path.exists(dataset_dir):
                raise ValueError(f"Dataset directory {os.path.abspath(dataset_dir)} does not exist")
            self.numpy_list = np.load(numpy_data)
        else:
            self.numpy_list = numpy_data

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
