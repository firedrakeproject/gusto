"""Common labels and routines for manipulating forms using labels."""

import ufl
from firedrake import Function
from firedrake.fml import Term, Label, LabelledForm
from gusto.core.configuration import IntegrateByParts, TransportEquationType
from types import MethodType

dynamics_label = Label("dynamics", validator=lambda value: type(value) is str)
physics_label = Label("physics", validator=lambda value: type(value) is str)


class DynamicsLabel(Label):
    """A label for a fluid dynamics term."""
    def __call__(self, target, value=None):
        """
        Applies the label to a form or term, and additionally labels the term as
        a dynamics term.

        Args:
            target (:class:`ufl.Form`, :class:`Term` or :class:`LabelledForm`):
                the form, term or labelled form to be labelled.
            value (..., optional): the value to attach to this label. Defaults
                to None.

        Returns:
            :class:`Term` or :class:`LabelledForm`: a :class:`Term` is returned
                if the target is a :class:`Term`, otherwise a
                :class:`LabelledForm` is returned.
        """

        new_target = dynamics_label(target, self.label)

        if isinstance(new_target, LabelledForm):
            # Need to be very careful in using super().__call__ method as the
            # underlying __call__ method calls itself to act upon multiple terms
            # in a LabelledForm. We can avoid this by special handling of the
            # LabelledForm case
            labelled_terms = (Label.__call__(self, t, value) for t in new_target.terms)
            return LabelledForm(*labelled_terms)
        else:
            new = super().__call__(new_target, value)
            return new


class PhysicsLabel(Label):
    """A label for a physics parametrisation term."""
    def __init__(self, label, *, value=True, validator=lambda value: type(value) == MethodType):
        """
        Args:
            label (str): the name of the label.
            value (..., optional): the value for the label to take. Can be any
                type (subject to the validator). Defaults to True.
            validator (func, optional): function to check the validity of any
                value later passed to the label. Defaults to None.
        """
        super().__init__(label, value=value, validator=validator)

    def __call__(self, target, value=None):
        """
        Applies the label to a form or term, and additionally labels the term as
        a physics term.

        Args:
            target (:class:`ufl.Form`, :class:`Term` or :class:`LabelledForm`):
                the form, term or labelled form to be labelled.
            value (..., optional): the value to attach to this label. Defaults
                to None.

        Returns:
            :class:`Term` or :class:`LabelledForm`: a :class:`Term` is returned
                if the target is a :class:`Term`, otherwise a
                :class:`LabelledForm` is returned.
        """

        new_target = physics_label(target, self.label)

        if isinstance(new_target, LabelledForm):
            # Need to be very careful in using super().__call__ method as the
            # underlying __call__ method calls itself to act upon multiple terms
            # in a LabelledForm. We can avoid this by special handling of the
            # LabelledForm case
            labelled_terms = (Label.__call__(self, t, value) for t in new_target.terms)
            return LabelledForm(*labelled_terms)
        else:
            new = super().__call__(new_target, value)
            return new


# ---------------------------------------------------------------------------- #
# Common Labels
# ---------------------------------------------------------------------------- #
implicit = Label("implicit")
explicit = Label("explicit")
horizontal_transport = Label("horizontal_transport")
vertical_transport = Label("vertical_transport")
source_label = Label("source_label")
transporting_velocity = Label("transporting_velocity", validator=lambda value: type(value) in [Function, ufl.tensors.ListTensor, ufl.indexed.Indexed])
prognostic = Label("prognostic", validator=lambda value: type(value) == str)
linearisation = Label("linearisation", validator=lambda value: type(value) in [LabelledForm, Term])
mass_weighted = Label("mass_weighted", validator=lambda value: type(value) in [LabelledForm, Term])
ibp_label = Label("ibp", validator=lambda value: type(value) == IntegrateByParts)
all_but_last = Label("all_but_last", validator=lambda value: type(value) in [LabelledForm, Term])


# labels for terms in the equations
time_derivative = Label("time_derivative")
nonlinear_time_derivative = Label("nonlinear_time_derivative")
transport = Label("transport",
                  validator=lambda value: type(value) == TransportEquationType)
diffusion = Label("diffusion")
pressure_gradient = DynamicsLabel("pressure_gradient")
coriolis = DynamicsLabel("coriolis")
divergence = DynamicsLabel("divergence")
gravity = DynamicsLabel("gravity")
hydrostatic = DynamicsLabel("hydrostatic")
incompressible = DynamicsLabel("incompressible")
sponge = DynamicsLabel("sponge")
