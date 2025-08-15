from abc import ABCMeta, abstractmethod, abstractproperty
from gusto.core import (Domain, IO, SubcyclingOptions, EmbeddedDGOptions,
                        SUPGOptions)
from gusto.recovery import BoundaryMethod, RecoverySpaces
from gusto.spatial_methods import (DGUpwind, InteriorPenaltyDiffusion,
                                   ThetaLimiter, DefaultTransport)
from gusto.time_discretisation import (SSPRK3, RungeKuttaFormulation,
                                       BackwardEuler, TrapeziumRule,
                                       ForwardEuler)
from gusto.timestepping import SemiImplicitQuasiNewton, SplitPhysicsTimestepper


class ModelBase(object, metaclass=ABCMeta):
    """Base model class."""

    def __init__(self, mesh, dt, parameters, equations, output,
                 family=None, element_order=None,
                 no_normal_flow_bc_ids=None,
                 diffused_fields=None,
                 physics_parameterisations=None,
                 diagnostic_fields=None):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
            dt (float): the model timestep.
            parameters (:class:`EquationParameters`): class storing the model
                equation parameters.
            equations (:class:`PrognosticEquationSet`): defines the model's
                prognostic equations
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
            diffused_fields (iter, optional): in iterable of tuples
                (str, :class:`DiffusionParameters`) that specifies the
                fields to be diffused and their parameters (e.g. diffusion
                coefficients). Defaults to None.
            physics_parameterisations (iter, optional): an iterable of
                :class:`PhysicsParametrisation` objects that describe physical
                parametrisations to be included to add to the equation.
                Defaults to None.
            diagnostic_fields (iter, optional): an iterable of `DiagnosticField`
                objects. Defaults to None.

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
        self.eqns = equations(self.domain, parameters,
                              no_normal_flow_bc_ids=no_normal_flow_bc_ids)

        # set up I/O
        self.io = IO(self.domain, output, diagnostic_fields=diagnostic_fields)

        # store diffused fields
        self.diffused_fields = diffused_fields

        # store physics parameterisations
        self.physics_parameterisations = physics_parameterisations

        # set up timestepper
        self.setup_timestepper()

    def setup_diffused_fields(self):

        raise NotImplementedError('Please implement diffused fields with model class!')

    def setup_physics(self):

        raise NotImplementedError('Please implement physics schemes with model class!')

    @abstractmethod
    def setup_timestepper(self):
        # Set up the model timestepper. Must be implemented in the child class.
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


class TimestepperModel(ModelBase):

    def setup_timestepper(self):
        pass

class SIQNModel(ModelBase):
    """
    Base class for models using the semi implicit quasi Newton timestepper.
    """

    @abstractproperty
    def transport_methods(self):
        pass

    @abstractproperty
    def transport_schemes(self):
        pass

    @abstractproperty
    def diffusion_methods(self):
        pass

    @abstractproperty
    def diffusion_schemes(self):
        pass

    def setup_timestepper(self):

        # to be removed
        if self.linear_solver is not None:
            linear_solver = self.linear_solver(self.eqns)
        else:
            linear_solver = None

        # Time stepper
        self.stepper = SemiImplicitQuasiNewton(
            self.eqns, self.io, self.transport_schemes,
            spatial_methods=self.transport_methods+self.diffusion_methods,
            linear_solver=linear_solver,
            diffusion_schemes=self.diffusion_schemes,
            physics_schemes=self.physics_schemes
        )


class Model(SIQNModel):

    def __init__(self, mesh, dt, parameters, equations, output,
                 family=None,
                 # next line args are just while reproducing
                 linear_solver=None, supg=False,
                 no_normal_flow_bc_ids=None,
                 physics_schemes=None, diffused_fields=None,
                 diagnostic_fields=None):

        # JS: I don't like this hardcoded list - we could check the
        # transport label of each field's transport term (see lines
        # 47-50 of transport_methods.py and the __init__ method of the
        # SpatialMethods class) but that's quite a lot of code to get
        # at this label and I'm also nervous that the form may be changed...
        self.conservative_form_fields = ["D", "rho"]

        self.linear_solver = linear_solver

        self.supg = supg
        if supg:
            self.theta_opts = SUPGOptions()
        else:
            self.theta_opts = EmbeddedDGOptions()

        self.diffused_fields = diffused_fields

        super().__init__(mesh=mesh, dt=dt, parameters=parameters,
                         equations=equations, output=output,
                         family=family, element_order=1,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         physics_schemes=physics_schemes,
                         diagnostic_fields=diagnostic_fields)

    @property
    def transport_methods(self):

        transport_methods = []
        for field_name in self.eqns.field_names:
            if field_name in self.conservative_form_fields:
                transport_methods.append(
                    DGUpwind(self.eqns, field_name, advective_then_flux=True)
                )
            elif self.eqns.space_names[field_name] == "theta" and self.supg:
                transport_methods.append(
                    DGUpwind(self.eqns, field_name, ibp=self.theta_opts.ibp)
                )
            else:
                transport_methods.append(
                    DGUpwind(self.eqns, field_name)
                )
        return transport_methods

    @property
    def transport_schemes(self):

        subcycling_options = SubcyclingOptions(subcycle_by_courant=0.25)

        transport_schemes = []
        for field_name in self.eqns.field_names:
            if field_name in self.conservative_form_fields:
                transport_schemes.append(
                    SSPRK3(self.domain, field_name,
                           subcycling_options=subcycling_options,
                           rk_formulation=RungeKuttaFormulation.linear)
                )
            elif self.eqns.space_names[field_name] == "theta":
                transport_schemes.append(
                    SSPRK3(self.domain, field_name,
                           subcycling_options=subcycling_options,
                           options=self.theta_opts,
                           limiter=ThetaLimiter(self.domain.spaces("theta")))
                )
            else:
                transport_schemes.append(
                    SSPRK3(self.domain, field_name,
                           subcycling_options=subcycling_options)
                )
        return transport_schemes

    @property
    def diffusion_methods(self):

        diffusion_methods = []
        for (field_name, params) in self.diffused_fields:
            diffusion_methods.append(
                InteriorPenaltyDiffusion(self.eqns, field_name, params)
            )

        return diffusion_methods

    @property
    def diffusion_schemes(self):

        diffusion_schemes = []
        for (field_name, params) in self.diffused_fields:
            diffusion_schemes.append(
                BackwardEuler(self.domain, field_name)
            )

        return diffusion_schemes


class LowestOrderModel(SIQNModel):

    def __init__(self, mesh, dt, parameters, equations, output,
                 family=None,
                 linear_solver=None,
                 no_normal_flow_bc_ids=None,
                 physics_schemes=None, diffusion_schemes=None,
                 diagnostic_fields=None):

        # to be removed
        self.linear_solver=linear_solver

        super().__init__(mesh=mesh, dt=dt, parameters=parameters,
                         equations=equations, output=output,
                         family=family,
                         element_order=0,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         physics_schemes=physics_schemes,
                         diagnostic_fields=diagnostic_fields)

    @property
    def transport_methods(self):

        transport_methods = []
        for field_name in self.eqns.field_names:
            transport_methods.append(
                DGUpwind(self.eqns, field_name)
            )
        return transport_methods

    @property
    def transport_schemes(self):
        # Transport schemes -- set up options for using recovery wrapper
        boundary_methods = {'DG': BoundaryMethod.taylor,
                            'HDiv': BoundaryMethod.taylor}

        recovery_spaces = RecoverySpaces(
            self.domain, boundary_methods, use_vector_spaces=True
        )

        opts = {"HDiv": recovery_spaces.HDiv_options,
                "L2": recovery_spaces.DG_options,
                "theta": recovery_spaces.theta_options}

        transport_schemes = []
        for field_name in self.eqns.field_names:
            transport_schemes.append(
                SSPRK3(self.domain, field_name,
                       options=opts[self.eqns.space_names[field_name]])
            )

        return transport_schemes

    @property
    def diffusion_methods(self):
        return []

    @property
    def diffusion_schemes(self):
        return []


class OldDefaultModel(Model):

    @property
    def transport_methods(self):

        transport_methods = []
        for field_name in self.eqns.field_names:
            transport_methods.append(
                DGUpwind(self.eqns, field_name)
            )
        return transport_methods

    @property
    def transport_schemes(self):

        transport_schemes = []
        for field_name in self.eqns.field_names:
            if field_name == "u":
                transport_schemes.append(
                    TrapeziumRule(self.domain, field_name)
                )
            elif self.eqns.space_names[field_name] == "theta":
                transport_schemes.append(
                    SSPRK3(self.domain, field_name,
                           limiter=ThetaLimiter(self.domain.spaces("theta")))
                )
            else:
                transport_schemes.append(
                    SSPRK3(self.domain, field_name,
                           subcycling_options=SubcyclingOptions(fixed_subcycles=2))
                )
        return transport_schemes


class SplitPhysicsModel(ModelBase):

    def setup_timestepper(self):

        # Time stepper
        self.stepper = SplitPhysicsTimestepper(
            self.eqns, self.scheme, self.io, self.spatial_methods,
            self.physics_schemes
        )


class LinearModel(Model):

    def __init__(self, mesh, dt, parameters, equations, output,
                 family=None,
                 linear_solver=None,
                 no_normal_flow_bc_ids=None,
                 physics_schemes=None, diffused_fields=None,
                 diagnostic_fields=None):

        self.conservative_form_fields = ["D", "rho"]

        super().__init__(mesh=mesh, dt=dt, parameters=parameters,
                         equations=equations, output=output,
                         family=family,
                         linear_solver=linear_solver,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         physics_schemes=physics_schemes,
                         diffused_fields=diffused_fields,
                         diagnostic_fields=diagnostic_fields)

    @property
    def transport_methods(self):

        transport_methods = []
        for field_name in self.eqns.field_names:
            if field_name in self.conservative_form_fields:
                transport_methods.append(
                    DefaultTransport(self.eqns, field_name)
                )
            elif field_name != "u":
                transport_methods.append(
                    DGUpwind(self.eqns, field_name)
                )
        return transport_methods

    @property
    def transport_schemes(self):
        # Transport schemes
        transport_schemes = []
        for field_name in self.eqns.field_names:
            if field_name != "u":
                transport_schemes.append(
                    ForwardEuler(self.domain, field_name)
                )
        return transport_schemes

    @property
    def diffusion_methods(self):
        return []

    @property
    def diffusion_schemes(self):
        return []
