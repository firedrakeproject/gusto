"""
Wrappers are objects that wrap around particular time discretisations, applying
some generic operation before and after a standard time discretisation is
called.
"""

from abc import ABCMeta, abstractmethod
from firedrake import (
    FunctionSpace, Function, BrokenElement, Projector, Constant, as_ufl, dot, grad,
    TestFunction, MixedFunctionSpace, assemble
)
from firedrake.__future__ import interpolate
from firedrake.fml import Term
from gusto.core.configuration import EmbeddedDGOptions, RecoveryOptions, SUPGOptions
from gusto.recovery import Recoverer, ReversibleRecoverer, ConservativeRecoverer
from gusto.core.labels import transporting_velocity
from gusto.core.conservative_projection import ConservativeProjector
from gusto.core.function_spaces import is_cg
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
        self.original_space = None
        self.is_conservative = False

    @abstractmethod
    def setup(self, original_space):
        """
        Store the original function space of the prognostic variable.

        Within each child wrapper, setup performs standard set up routines,
        and is to be called by the setup method of the underlying
        time discretisation.

        Args:
            original_space (:class:`FunctionSpace`): the space that the
                prognostic variable is defined on. This is a subset space of
                a mixed function space when using a MixedFSWrapper.
        """
        self.original_space = original_space

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

    def setup(self, original_space, post_apply_bcs):
        """
        Sets up function spaces and fields needed for this wrapper.

        Args:
            original_space (:class:`FunctionSpace`): the space that the
                prognostic variable is defined on.
            post_apply_bcs (list of :class:`DirichletBC`): list of Dirichlet
                boundary condition objects to be passed to the projector used
                in the post-apply step.
        """

        assert isinstance(self.options, EmbeddedDGOptions), \
            'Embedded DG wrapper can only be used with Embedded DG Options'

        super().setup(original_space)

        domain = self.time_discretisation.domain
        equation = self.time_discretisation.equation

        # -------------------------------------------------------------------- #
        # Set up spaces to be used with wrapper
        # -------------------------------------------------------------------- #

        if self.options.embedding_space is None:
            V_elt = BrokenElement(self.original_space.ufl_element())
            self.function_space = FunctionSpace(domain.mesh, V_elt)
        else:
            self.function_space = self.options.embedding_space

        self.test_space = self.function_space

        # -------------------------------------------------------------------- #
        # Internal variables to be used
        # -------------------------------------------------------------------- #

        self.x_in = Function(self.function_space)
        self.x_out = Function(self.function_space)
        self.x_in_orig = Function(original_space)

        if self.time_discretisation.idx is None:
            self.x_projected = Function(self.original_space)
        else:
            self.x_projected = Function(equation.spaces[self.time_discretisation.idx])

        if self.options.project_back_method == 'project':
            self.x_out_projector = Projector(self.x_out, self.x_projected,
                                             bcs=post_apply_bcs)
        elif self.options.project_back_method == 'recover':
            self.x_out_projector = Recoverer(self.x_out, self.x_projected)
        elif self.options.project_back_method == 'conservative_project':
            self.is_conservative = True
            self.rho_name = self.options.rho_name
            self.rho_in_orig = Function(self.options.orig_rho_space)
            self.rho_out_orig = Function(self.options.orig_rho_space)
            self.rho_in_embedded = Function(self.function_space)
            self.rho_out_embedded = Function(self.function_space)
            self.x_in_projector = ConservativeProjector(
                self.rho_in_orig, self.rho_in_embedded,
                self.x_in_orig, self.x_in)
            self.x_out_projector = ConservativeProjector(
                self.rho_out_embedded, self.rho_out_orig,
                self.x_out, self.x_projected, subtract_mean=True)
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

        self.x_in_orig.assign(x_in)

        if self.is_conservative:
            self.x_in_projector.project()
        else:
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

    def setup(self, original_space, post_apply_bcs):
        """
        Sets up function spaces and fields needed for this wrapper.

        Args:
            original_space (:class:`FunctionSpace`): the space that the
                prognostic variable is defined on.
            post_apply_bcs (list of :class:`DirichletBC`): list of Dirichlet
                boundary condition objects to be passed to the projector used
                in the post-apply step.
        """

        assert isinstance(self.options, RecoveryOptions), \
            'Recovery wrapper can only be used with Recovery Options'

        super().setup(original_space)

        domain = self.time_discretisation.domain
        equation = self.time_discretisation.equation

        # -------------------------------------------------------------------- #
        # Set up spaces to be used with wrapper
        # -------------------------------------------------------------------- #

        if self.options.embedding_space is None:
            V_elt = BrokenElement(self.original_space.ufl_element())
            self.function_space = FunctionSpace(domain.mesh, V_elt)
        else:
            self.function_space = self.options.embedding_space

        self.test_space = self.function_space

        # -------------------------------------------------------------------- #
        # Internal variables to be used
        # -------------------------------------------------------------------- #

        self.x_in_orig = Function(self.original_space)
        self.x_in = Function(self.function_space)
        self.x_out = Function(self.function_space)

        if self.time_discretisation.idx is None:
            self.x_projected = Function(self.original_space)
        else:
            self.x_projected = Function(equation.spaces[self.time_discretisation.idx])

        # Operator to recover to higher discontinuous space
        if self.options.project_low_method == 'conservative_project':
            self.is_conservative = True
            self.rho_name = self.options.rho_name
            self.rho_in_orig = Function(self.options.orig_rho_space)
            self.rho_out_orig = Function(self.options.orig_rho_space)
            self.rho_in_embedded = Function(self.function_space)
            self.rho_out_embedded = Function(self.function_space)
            self.x_recoverer = ConservativeRecoverer(self.x_in_orig, self.x_in,
                                                     self.rho_in_orig,
                                                     self.rho_in_embedded,
                                                     self.options)
        else:
            self.x_recoverer = ReversibleRecoverer(self.x_in_orig, self.x_in, self.options)

        # Operators for projecting back
        self.interp_back = (self.options.project_low_method == 'interpolate')
        if self.options.project_low_method == 'interpolate':
            self.x_out_projector = interpolate(self.x_out, self.original_space)
        elif self.options.project_low_method == 'project':
            self.x_out_projector = Projector(self.x_out, self.x_projected,
                                             bcs=post_apply_bcs)
        elif self.options.project_low_method == 'recover':
            self.x_out_projector = Recoverer(self.x_out, self.x_projected,
                                             method=self.options.broken_method)
        elif self.options.project_low_method == 'conservative_project':
            self.x_out_projector = ConservativeProjector(
                self.rho_out_embedded, self.rho_out_orig,
                self.x_out, self.x_projected, subtract_mean=True)
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

        self.x_in_orig.assign(x_in)
        self.x_recoverer.project()

    def post_apply(self, x_out):
        """
        Extra post-apply steps for the recovered method. Projects the output
        field from the embedding space to the original space.

        Args:
            x_out (:class:`Function`): the output field in the original space.
        """

        if self.interp_back:
            self.x_projected.assign(assemble(self.x_out_projector))
        else:
            self.x_out_projector.project()
        x_out.assign(self.x_projected)


class SUPGWrapper(Wrapper):
    """
    Wrapper for computing a time discretisation with SUPG, which adjusts the
    test function space that is used to solve the problem.
    """

    def setup(self, field_name):
        """Sets up function spaces and fields needed for this wrapper."""

        assert isinstance(self.options, SUPGOptions), \
            'SUPG wrapper can only be used with SUPG Options'

        domain = self.time_discretisation.domain
        if self.options.suboptions is not None:
            self.idx = self.time_discretisation.equation.field_names.index(field_name)
            self.test_space = self.time_discretisation.equation.spaces[self.idx]
        else:
            self.idx = None
            self.test_space = self.time_discretisation.fs
        self.function_space = self.time_discretisation.fs
        self.x_out = Function(self.function_space)
        self.field_name = field_name

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
            if is_cg(self.test_space):
                vals = default_vals
            else:
                space = self.test_space.ufl_element().sobolev_space
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
        if self.options.suboptions is not None:
            test = self.time_discretisation.equation.tests[self.idx]
        else:
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
        self.is_conservative = any([subwrapper.is_conservative for subwrapper in self.subwrappers.values()])

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
                if subwrapper.is_conservative:
                    self.pre_update_rho(subwrapper)
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
                if subwrapper.is_conservative:
                    self.post_update_rho(subwrapper)
                subwrapper.post_apply(x_out_sub)
            else:
                x_out_sub.assign(field)

    def pre_update_rho(self, subwrapper):
        """
        Updates the stored density field for the pre-apply for the subwrapper.

        Args:
            subwrapper (:class:`Wrapper`): the original input field.
        """

        rho_subwrapper = self.subwrappers[subwrapper.rho_name]

        subwrapper.rho_in_orig.assign(rho_subwrapper.x_in_orig)
        subwrapper.rho_in_embedded.assign(rho_subwrapper.x_in)

    def post_update_rho(self, subwrapper):
        """
        Updates the stored density field for the post-apply for the subwrapper.

        Args:
            subwrapper (:class:`Wrapper`): the original input field.
        """

        rho_subwrapper = self.subwrappers[subwrapper.rho_name]

        subwrapper.rho_out_orig.assign(rho_subwrapper.x_projected)
        subwrapper.rho_out_embedded.assign(rho_subwrapper.x_out)
