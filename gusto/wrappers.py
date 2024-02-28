"""
Wrappers are objects that wrap around particular time discretisations, applying
some generic operation before and after a standard time discretisation is
called.
"""

from abc import ABCMeta, abstractmethod
from firedrake import (
    FunctionSpace, Function, BrokenElement, Projector, Interpolator,
    VectorElement, Constant, as_ufl, dot, grad, TestFunction, MixedFunctionSpace
)
from firedrake.fml import Term
from gusto.configuration import EmbeddedDGOptions, RecoveryOptions, SUPGOptions
from gusto.recovery import Recoverer, ReversibleRecoverer
from gusto.labels import transporting_velocity
import ufl

__all__ = ["EmbeddedDGWrapper", "RecoveryWrapper", "SUPGWrapper", "MixedFSWrapper"]


class Wrapper(object, metaclass=ABCMeta):
    """Base class for time discretisation wrapper objects."""

    def __init__(self, time_discretisation, wrapper_options):
        """
        Args:
            time_discretisation (:class:`TimeDiscretisation`): the time
                discretisation that this wrapper is to be used with.
            wrapper_options (:class:`WrapperOptions`): configuration object
                holding the options specific to this `Wrapper`.
        """

        self.time_discretisation = time_discretisation
        self.options = wrapper_options
        self.solver_parameters = None

    @abstractmethod
    def setup(self):
        """
        Performs standard set up routines, and is to be called by the setup
        method of the underlying time discretisation.
        """
        pass

    @abstractmethod
    def pre_apply(self):
        """Generic steps to be done before time discretisation apply method."""
        pass

    @abstractmethod
    def post_apply(self):
        """Generic steps to be done after time discretisation apply method."""
        pass

    def label_terms(self, residual):
        """
        A base method to allow labels to be updated or extra labels added to
        the form that will be used with the wrapper. This base method does
        nothing but there may be implementations in child classes.

        Args:
            residual (:class:`LabelledForm`): the labelled form to update.

        Returns:
            :class:`LabelledForm`: the updated labelled form.
        """

        return residual


class EmbeddedDGWrapper(Wrapper):
    """
    Wrapper for computing a time discretisation with the Embedded DG method, in
    which a field is converted to an embedding discontinuous space, then evolved
    using a method suitable for this space, before projecting the field back to
    the original space.
    """

    def setup(self, previous_space=None):
        """Sets up function spaces and fields needed for this wrapper."""

        assert isinstance(self.options, EmbeddedDGOptions), \
            'Embedded DG wrapper can only be used with Embedded DG Options'

        domain = self.time_discretisation.domain
        equation = self.time_discretisation.equation

        if previous_space is not None:
            original_space = previous_space
        else:
            original_space = self.time_discretisation.fs

        # -------------------------------------------------------------------- #
        # Set up spaces to be used with wrapper
        # -------------------------------------------------------------------- #

        if self.options.embedding_space is None:
            V_elt = BrokenElement(original_space.ufl_element())
            self.function_space = FunctionSpace(domain.mesh, V_elt)
        else:
            self.function_space = self.options.embedding_space

        self.test_space = self.function_space

        # -------------------------------------------------------------------- #
        # Internal variables to be used
        # -------------------------------------------------------------------- #

        self.x_in = Function(self.function_space)
        self.x_out = Function(self.function_space)

        if previous_space is not None:
            self.x_projected = Function(previous_space)
        elif self.time_discretisation.idx is None:
            self.x_projected = Function(equation.function_space)
        else:
            self.x_projected = Function(equation.spaces[self.time_discretisation.idx])

        if self.options.project_back_method == 'project':
            self.x_out_projector = Projector(self.x_out, self.x_projected)
        elif self.options.project_back_method == 'recover':
            self.x_out_projector = Recoverer(self.x_out, self.x_projected)
        else:
            raise NotImplementedError(
                'EmbeddedDG Wrapper: project_back_method'
                + f' {self.options.project_back_method} is not implemented')

        self.parameters = {'ksp_type': 'cg',
                           'pc_type': 'bjacobi',
                           'sub_pc_type': 'ilu'}

    def pre_apply(self, x_in):
        """
        Extra pre-apply steps for the embedded DG method. Interpolates or
        projects x_in to the embedding space.

        Args:
            x_in (:class:`Function`): the original input field.
        """

        try:
            self.x_in.interpolate(x_in)
        except NotImplementedError:
            self.x_in.project(x_in)

    def post_apply(self, x_out):
        """
        Extra post-apply steps for the embedded DG method. Projects the output
        field from the embedding space to the original space.

        Args:
            x_out (:class:`Function`): the output field in the original space.
        """

        self.x_out_projector.project()
        x_out.assign(self.x_projected)


class RecoveryWrapper(Wrapper):
    """
    Wrapper for computing a time discretisation with the "recovered" method, in
    which a field is converted to higher-order function space space. The field
    is then evolved in this higher-order function space to obtain an increased
    order of accuracy over evolving the field in the lower-order space. The
    field is then returned to the original space.
    """

    def setup(self, previous_space=None):
        """Sets up function spaces and fields needed for this wrapper."""

        assert isinstance(self.options, RecoveryOptions), \
            'Recovery wrapper can only be used with Recovery Options'

        domain = self.time_discretisation.domain
        equation = self.time_discretisation.equation

        if previous_space is not None:
            original_space = previous_space
        else:
            original_space = self.time_discretisation.fs

        # -------------------------------------------------------------------- #
        # Set up spaces to be used with wrapper
        # -------------------------------------------------------------------- #

        if self.options.embedding_space is None:
            V_elt = BrokenElement(original_space.ufl_element())
            self.function_space = FunctionSpace(domain.mesh, V_elt)
        else:
            self.function_space = self.options.embedding_space

        self.test_space = self.function_space

        # -------------------------------------------------------------------- #
        # Internal variables to be used
        # -------------------------------------------------------------------- #

        if previous_space is not None:
            self.x_in_tmp = Function(previous_space)
        else:
            self.x_in_tmp = Function(self.time_discretisation.fs)

        self.x_in = Function(self.function_space)
        self.x_out = Function(self.function_space)

        if previous_space is not None:
            self.x_projected = Function(previous_space)
        elif self.time_discretisation.idx is None:
            self.x_projected = Function(equation.function_space)
        else:
            self.x_projected = Function(equation.spaces[self.time_discretisation.idx])

        # Operator to recover to higher discontinuous space
        self.x_recoverer = ReversibleRecoverer(self.x_in_tmp, self.x_in, self.options)

        # Operators for projecting back
        self.interp_back = (self.options.project_low_method == 'interpolate')
        if self.options.project_low_method == 'interpolate':
            self.x_out_projector = Interpolator(self.x_out, self.x_projected)
        elif self.options.project_low_method == 'project':
            self.x_out_projector = Projector(self.x_out, self.x_projected)
        elif self.options.project_low_method == 'recover':
            self.x_out_projector = Recoverer(self.x_out, self.x_projected,
                                             method=self.options.broken_method)
        else:
            raise NotImplementedError(
                'Recovery Wrapper: project_back_method'
                + f' {self.options.project_back_method} is not implemented')

    def pre_apply(self, x_in):
        """
        Extra pre-apply steps for the recovered method. Interpolates or projects
        x_in to the embedding space.

        Args:
            x_in (:class:`Function`): the original input field.
        """

        self.x_in_tmp.assign(x_in)
        self.x_recoverer.project()

    def post_apply(self, x_out):
        """
        Extra post-apply steps for the recovered method. Projects the output
        field from the embedding space to the original space.

        Args:
            x_out (:class:`Function`): the output field in the original space.
        """

        if self.interp_back:
            self.x_out_projector.interpolate()
        else:
            self.x_out_projector.project()
        x_out.assign(self.x_projected)


def is_cg(V):
    """
    Checks if a :class:`FunctionSpace` is continuous.

    Function to check if a given space, V, is CG. Broken elements are always
    discontinuous; for vector elements we check the names of the Sobolev spaces
    of the subelements and for all other elements we just check the Sobolev
    space name.

    Args:
        V (:class:`FunctionSpace`): the space to check.
    """
    ele = V.ufl_element()
    if isinstance(ele, BrokenElement):
        return False
    elif type(ele) == VectorElement:
        return all([e.sobolev_space.name == "H1" for e in ele._sub_elements])
    else:
        return V.ufl_element().sobolev_space.name == "H1"


class SUPGWrapper(Wrapper):
    """
    Wrapper for computing a time discretisation with SUPG, which adjusts the
    test function space that is used to solve the problem.
    """

    def setup(self):
        """Sets up function spaces and fields needed for this wrapper."""

        assert isinstance(self.options, SUPGOptions), \
            'SUPG wrapper can only be used with SUPG Options'

        domain = self.time_discretisation.domain
        self.function_space = self.time_discretisation.fs
        self.test_space = self.function_space
        self.x_out = Function(self.function_space)

        # -------------------------------------------------------------------- #
        # Work out SUPG parameter
        # -------------------------------------------------------------------- #

        # construct tau, if it is not specified
        dim = domain.mesh.topological_dimension()
        if self.options.tau is not None:
            # if tau is provided, check that is has the right size
            self.tau = self.options.tau
            assert as_ufl(self.tau).ufl_shape == (dim, dim), "Provided tau has incorrect shape!"
        else:
            # create tuple of default values of size dim
            default_vals = [self.options.default*self.time_discretisation.dt]*dim
            # check for directions is which the space is discontinuous
            # so that we don't apply supg in that direction
            if is_cg(self.function_space):
                vals = default_vals
            else:
                space = self.function_space.ufl_element().sobolev_space
                if space.name in ["HDiv", "DirectionalH"]:
                    vals = [default_vals[i] if space[i].name == "H1"
                            else 0. for i in range(dim)]
                else:
                    raise ValueError("I don't know what to do with space %s" % space)
            self.tau = Constant(tuple([
                tuple(
                    [vals[j] if i == j else 0. for i, v in enumerate(vals)]
                ) for j in range(dim)])
            )
            self.solver_parameters = {'ksp_type': 'gmres',
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}

        # -------------------------------------------------------------------- #
        # Set up test function
        # -------------------------------------------------------------------- #

        test = TestFunction(self.test_space)
        uadv = Function(domain.spaces('HDiv'))
        self.test = test + dot(dot(uadv, self.tau), grad(test))
        self.transporting_velocity = uadv

    def pre_apply(self, x_in):
        """
        Does nothing for SUPG, just sets the input field.

        Args:
            x_in (:class:`Function`): the original input field.
        """

        self.x_in = x_in

    def post_apply(self, x_out):
        """
        Does nothing for SUPG, just sets the output field.

        Args:
            x_out (:class:`Function`): the output field in the original space.
        """

        x_out.assign(self.x_out)

    def label_terms(self, residual):
        """
        A base method to allow labels to be updated or extra labels added to
        the form that will be used with the wrapper.

        Args:
            residual (:class:`LabelledForm`): the labelled form to update.

        Returns:
            :class:`LabelledForm`: the updated labelled form.
        """

        new_residual = residual.label_map(
            lambda t: t.has_label(transporting_velocity),
            # Update and replace transporting velocity in any terms
            map_if_true=lambda t:
            Term(ufl.replace(t.form, {t.get(transporting_velocity): self.transporting_velocity}), t.labels),
            # Add new label to other terms
            map_if_false=lambda t: transporting_velocity(t, self.transporting_velocity)
        )

        new_residual = transporting_velocity.update_value(new_residual, self.transporting_velocity)

        return new_residual


class MixedFSWrapper(object):
    """
    An object to hold a subwrapper dictionary with different wrappers for
    different tracers. This means that different tracers can be solved
    simultaneously using a CoupledTransportEquation, whilst being in
    different spaces and needing different implementation options.
    """

    def __init__(self):

        self.wrapper_spaces = None
        self.field_names = None
        self.subwrappers = {}

    def setup(self):
        """ Compute the new mixed function space from the subwrappers """

        self.function_space = MixedFunctionSpace(self.wrapper_spaces)
        self.x_in = Function(self.function_space)
        self.x_out = Function(self.function_space)

    def pre_apply(self, x_in):
        """
        Perform the pre-applications for all fields
        with an associated subwrapper.
        """

        for field_name in self.field_names:
            field_idx = self.field_names.index(field_name)
            field = x_in.subfunctions[field_idx]
            x_in_sub = self.x_in.subfunctions[field_idx]

            if field_name in self.subwrappers:
                subwrapper = self.subwrappers[field_name]
                subwrapper.pre_apply(field)
                x_in_sub.assign(subwrapper.x_in)
            else:
                x_in_sub.assign(field)

    def post_apply(self, x_out):
        """
        Perform the post-applications for all fields
        with an associated subwrapper.
        """

        for field_name in self.field_names:
            field_idx = self.field_names.index(field_name)
            field = self.x_out.subfunctions[field_idx]
            x_out_sub = x_out.subfunctions[field_idx]

            if field_name in self.subwrappers:
                subwrapper = self.subwrappers[field_name]
                subwrapper.x_out.assign(field)
                subwrapper.post_apply(x_out_sub)
            else:
                x_out_sub.assign(field)
