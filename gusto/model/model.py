from abc import ABCMeta, abstractmethod
from gusto.core import (Domain, IO, EmbeddedDGOptions)
from gusto.spatial_methods import DGUpwind
from gusto.time_discretisation import SSPRK3, RungeKuttaFormulation
from gusto.timestepping import SemiImplicitQuasiNewton


class ModelBase(object, metaclass=ABCMeta):
    """Base model class."""

    def __init__(self, mesh, dt, parameters, equation, output,
                 family=None, element_order=None,
                 no_normal_flow_bc_ids=None):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
            dt (float): the model timestep.
            parameters (:class:`EquationParameters`): class storing the model
                equation parameters.
            equation (:class:`PrognosticEquationSet`): defines the model's
                prognostic equation
            output (:class:`OutputParameters`): provides parameters
                controlling output
            family (str, optional): the finite element space family used for
                the velocity field. This determines the other finite element
                spaces used via the de Rham complex. If not provided, an
                appropriate choice will be made based on the cell of the mesh
                (or the base mesh in 3D).
            element_order (int): the element degree used for the DG
                space. Defaults to None.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
        """
        # if HDiv finite element family not provided figure out a
        # sensible default based on the cell shape of the mesh
        if family is None:
            extruded_mesh = hasattr(mesh, "_base_mesh")
            if extruded_mesh:
                cell = mesh._base_mesh.ufl_cell().cellname()
            else:
                cell = mesh.ufl_cell().cellname()
            if cell == "interval":
                family = "CG"
            elif cell == "quadrilateral":
                family = "RTCF"
            elif cell == "triangle":
                family = "BDM"
            else:
                raise ValueError(f"The mesh provided (or its base mesh if 3D) must have cells of type interval, quadrilateral or triangle, not {cell}.")

        # create domain
        self.domain = Domain(mesh, dt, family, element_order)

        # set up prognostic equations
        self.equation = equation(self.domain, parameters,
                                 no_normal_flow_bc_ids=no_normal_flow_bc_ids)

        # save output options for when IO is set up later
        self.output = output

    @abstractmethod
    def setup(self):
        # Set up the model. Must be implemented in the child class.
        pass

    def run(self, t, tmax, pick_up=False):
        """
        Runs the model for the specified time, from t to tmax

        Args:
            t (float): the start time of the run
            tmax (float): the end time of the run
            pick_up: (bool): specify whether to pick_up from a previous run
        """
        self.stepper.run(t=t, tmax=tmax, pick_up=pick_up)


class SIQNModel(ModelBase):

    @property
    def transported_fields(self):
        transported_fields = []
        for field_name in self.equation.field_names:
            if self.equation.space_names[field_name] == 'L2':
                transported_fields.append(
                    SSPRK3(self.domain, field_name,
                           rk_formulation=RungeKuttaFormulation.linear)
                )
            elif self.equation.space_names[field_name] == 'theta':
                transported_fields.append(
                    SSPRK3(self.domain, field_name,
                           options=EmbeddedDGOptions())
                )
            else:
                transported_fields.append(
                    SSPRK3(
                        self.domain, field_name)
                )
        return transported_fields

    @property
    def transport_methods(self):
        transport_methods = []
        for field_name in self.equation.field_names:
            if self.equation.space_names[field_name] == 'L2':
                transport_methods.append(
                    DGUpwind(self.equation, field_name,
                             advective_then_flux=True)
                )
            else:
                transport_methods.append(
                    DGUpwind(self.equation, field_name)
                )
        return transport_methods

    @property
    def tau_values(self):
        tau_values = {}
        for field_name in self.equation.field_names:
            if field_name is not "u":
                tau_values[field_name] = 1.0
        return tau_values

    def setup(self, diagnostic_fields):

        io = IO(self.domain, self.output, diagnostic_fields=diagnostic_fields)
        self.stepper = SemiImplicitQuasiNewton(
            self.equation, io, self.transported_fields,
            spatial_methods=self.transport_methods,
            tau_values=self.tau_values
        )
