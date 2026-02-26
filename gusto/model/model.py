from abc import ABCMeta, abstractmethod, abstractproperty
from gusto.core import (Domain, IO, EmbeddedDGOptions)
from gusto.recovery import BoundaryMethod, RecoverySpaces
from gusto.spatial_methods import DGUpwind, InteriorPenaltyDiffusion
from gusto.time_discretisation import (SSPRK3, RungeKuttaFormulation,
                                       BackwardEuler)
from gusto.timestepping import SemiImplicitQuasiNewton


class ModelBase(object, metaclass=ABCMeta):
    """Base model class."""

    def __init__(self, mesh, dt, parameters, equation,
                 element_order, family=None,
                 **kwargs):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
            dt (float): the model timestep.
            parameters (:class:`EquationParameters`): class storing the model
                equation parameters.
            equation (:class:`PrognosticEquationSet`): defines the model's
                prognostic equation
            element_order (int): the element degree used for the DG
                space. Defaults to None.
            family (str, optional): the finite element space family used for
                the velocity field. This determines the other finite element
                spaces used via the de Rham complex. If not provided, an
                appropriate choice will be made based on the cell of the mesh
                (or the base mesh in 3D).

        Kwargs:
            Kwargs are passed straight through to the equation class; see
            those classes for full documentation. Some common kwargs are
            listed below.

            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form', and
                'circulation_form'.Defaults to 'vector_invariant_form'.
            diffusion_options (iterable, optional): iterable of
                ``(field_name, diffusion_parameters)`` pairs where
                diffusion_parameters is a :class:`DiffusionParameters`
                object specifying the diffusion parameters to be applied
                to the field field_name.  Defaults to None.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. Defaults to None.
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
                raise ValueError(f"The mesh provided (or its base mesh if \
                extruded) must have cells of type interval, quadrilateral \
                or triangle, not {cell}.")

        # create domain
        self.domain = Domain(mesh, dt, family, element_order)

        # set up prognostic equations
        self.equation = equation(self.domain, parameters, **kwargs)

        # store diffusions options as needed to set up spatial methods
        # and diffusion schemes - default is an empty list
        self.diffusion_options = kwargs.get("diffusion_options", [])

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


class SIQNModelBase(ModelBase):
    """
    Base for model classes using SIQN. Child classes should define the
    standard transport and diffusion schemes and methods.
    """
    @abstractproperty
    def diffusion_methods(self):
        pass

    @abstractproperty
    def diffusion_schemes(self):
        pass

    @abstractproperty
    def transported_fields(self):
        pass

    @abstractproperty
    def transport_methods(self):
        pass

    @property
    def tau_values(self):
        _tau_values = {}
        for field_name in self.equation.field_names:
            if field_name != "u":
                _tau_values[field_name] = 1.0
        return _tau_values
        pass

    def setup(self, output, **kwargs):
        """
        Args:
            output (:class:`OutputParameters`): provides parameters
                controlling output

        Kwargs:
            diagnostic_fields
            subcycling_options
            Remaining kwargs are passed straight through to the stepper.
        """

        diagnostic_fields = kwargs.pop("diagnostic_fields", None)
        io = IO(self.domain, output, diagnostic_fields=diagnostic_fields)

        self.subcycling_options = kwargs.pop("subcycling_options", None)
        self.limiters = kwargs.pop("limiters", {})

        self.stepper = SemiImplicitQuasiNewton(
            self.equation, io, self.transported_fields,
            spatial_methods=self.transport_methods+self.diffusion_methods,
            diffusion_schemes=self.diffusion_schemes,
            tau_values=self.tau_values,
            **kwargs
        )


class SIQNModel(SIQNModelBase):
    """
    SIQN model class encapsulating the best settings for next-to-lowest
    order methods. Currently the standard Gusto model.
    """
    def __init__(self, mesh, dt, parameters, equation,
                 family=None,
                 no_normal_flow_bc_ids=None,
                 **kwargs):

        super().__init__(mesh, dt, parameters, equation,
                         family=family, element_order=1,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         **kwargs)

    @property
    def diffusion_methods(self):
        _diffusion_methods = []
        for field, params in self.diffusion_options:
            _diffusion_methods.append(
                InteriorPenaltyDiffusion(self.equation, field, params)
            )
        return _diffusion_methods

    @property
    def diffusion_schemes(self):
        _diffusion_schemes = []
        for field, _ in self.diffusion_options:
            _diffusion_schemes.append(
                BackwardEuler(self.domain, field)
            )
        return _diffusion_schemes

    @property
    def transported_fields(self):
        _transported_fields = []
        for field_name in self.equation.field_names:
            if self.equation.space_names[field_name] == 'L2':
                _transported_fields.append(
                    SSPRK3(self.domain, field_name,
                           subcycling_options=self.subcycling_options,
                           rk_formulation=RungeKuttaFormulation.linear,
                           limiter=self.limiters.get(field_name))
                )
            elif self.equation.space_names[field_name] == 'theta':
                _transported_fields.append(
                    SSPRK3(self.domain, field_name,
                           subcycling_options=self.subcycling_options,
                           options=EmbeddedDGOptions(),
                           limiter=self.limiters.get(field_name))
                )
            else:
                _transported_fields.append(
                    SSPRK3(
                        self.domain, field_name,
                        subcycling_options=self.subcycling_options,
                        limiter=self.limiters.get(field_name))
                )
        return _transported_fields

    @property
    def transport_methods(self):
        _transport_methods = []
        for field_name in self.equation.field_names:
            if self.equation.space_names[field_name] == 'L2':
                _transport_methods.append(
                    DGUpwind(self.equation, field_name,
                             advective_then_flux=True)
                )
            else:
                _transport_methods.append(
                    DGUpwind(self.equation, field_name)
                )
        return _transport_methods


class LowestOrderSIQNModel(SIQNModelBase):
    """
    SIQN model class encapsulating the best settings for lowest
    order methods.
    """
    def __init__(self, mesh, dt, parameters, equation,
                 family=None,
                 no_normal_flow_bc_ids=None, **kwargs):

        super().__init__(mesh, dt, parameters, equation,
                         family=family, element_order=0,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         **kwargs)

    @property
    def diffusion_methods(self):
        _diffusion_methods = []
        for field, params in self.diffusion_options:
            _diffusion_methods.append(
                InteriorPenaltyDiffusion(self.equation, field, params)
            )
        return _diffusion_methods

    @property
    def diffusion_schemes(self):
        _diffusion_schemes = []
        for field, _ in self.diffusion_options:
            _diffusion_schemes.append(
                BackwardEuler(self.domain, field)
            )
        return _diffusion_schemes

    @property
    def transported_fields(self):
        boundary_methods = {'DG': BoundaryMethod.taylor}
        recovery_spaces = RecoverySpaces(
            self.domain, boundary_methods, use_vector_spaces=True
        )
        _transported_fields = []
        for field_name in self.equation.field_names:
            if self.equation.space_names[field_name] == 'HDiv':
                _transported_fields.append(
                    SSPRK3(self.domain, field_name,
                           subcycling_options=self.subcycling_options,
                           options=recovery_spaces.HDiv_options,
                           limiter=self.limiters.get(field_name))
                )
            elif self.equation.space_names[field_name] == 'L2':
                _transported_fields.append(
                    SSPRK3(self.domain, field_name,
                           subcycling_options=self.subcycling_options,
                           options=recovery_spaces.DG_options,
                           limiter=self.limiters.get(field_name))
                )
            elif self.equation.space_names[field_name] == 'theta':
                _transported_fields.append(
                    SSPRK3(self.domain, field_name,
                           subcycling_options=self.subcycling_options,
                           options=recovery_spaces.theta_options,
                           limiter=self.limiters.get(field_name))
                )
        return _transported_fields

    @property
    def transport_methods(self):
        _transport_methods = []
        for field_name in self.equation.field_names:
            _transport_methods.append(DGUpwind(self.equation, field_name))
        return _transport_methods
