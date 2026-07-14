"""
A module defining objects for temporarily augmenting an equation with another.
"""


from abc import ABCMeta, abstractmethod
from firedrake import (
    MixedFunctionSpace, Function, TestFunctions, split, inner, dx, grad,
    LinearVariationalProblem, LinearVariationalSolver, lhs, rhs, dot,
    ds_b, ds_v, ds_t, ds, FacetNormal, TestFunction, TrialFunction,
    transpose, nabla_grad, outer, dS, dS_h, dS_v, sign, jump, div,
    Constant, sqrt, cross, curl, FunctionSpace, assemble, DirichletBC
)
from firedrake.fml import (
    subject, all_terms, replace_subject, replace_test_function,
    drop, Term, LabelledForm
)
from gusto import (
    time_derivative, transport, transporting_velocity, TransportEquationType,
    logger, prognostic, mass_weighted
)
from gusto.spatial_methods.limiters import MeanLimiter
from gusto.core.conservative_projection import ConservativeProjector
import numpy as np


class Augmentation(object, metaclass=ABCMeta):
    """
    Augments an equation with another equation to be solved simultaneously.
    """

    @abstractmethod
    def pre_apply(self, x_in):
        """
        Steps to take at the beginning of an apply method, for instance to
        assign the input field to the internal mixed function.
        """

        pass

    @abstractmethod
    def post_apply(self, x_out):
        """
        Steps to take at the end of an apply method, for instance to assign the
        internal mixed function to the output field.
        """

        pass

    @abstractmethod
    def update(self, x_in_mixed):
        """
        Any intermediate update steps, depending on the current mixed function.
        """

        pass

    def limit(self, x_in_mixed):
        """
        Apply any special limiting as part of the augmentation
        """

        pass


class VorticityTransport(Augmentation):
    """
    Solves the transport of a vector field, simultaneously with the vorticity
    as a mixed proble, as described in Bendall and Wimmer (2022).

    Note that this is most effective with implicit time discretisations. The
    residual-based SUPG option provides a dissipation method.

    Args:
        domain (:class:`Domain`): The domain object.
        eqns (:class:`PrognosticEquationSet`): The overarching equation set.
        transpose_commutator (bool, optional): Whether to include the commutator
            of the transpose gradient terms. This is necessary for solving the
            general vector transport equation, but is not necessary when the
            transporting and transported fields are the same. Defaults to True.
        supg (bool, optional): Whether to include dissipation through a
            residual-based SUPG scheme. Defaults to False.
    """

    def __init__(
            self, domain, eqns, transpose_commutator=True, supg=False
    ):

        self.name = 'vorticity'

        V_vel = domain.spaces('HDiv')
        V_vort = domain.spaces('H1')

        self.fs = MixedFunctionSpace((V_vel, V_vort))
        self.X = Function(self.fs)
        self.tests = TestFunctions(self.fs)

        u = Function(V_vel)
        F, Z = split(self.X)
        test_F, test_Z = self.tests

        quad = domain.max_quad_degree

        if domain.mesh.extruded:
            self.ds = ds_b(degree=quad) + ds_t(degree=quad) + ds_v(degree=quad)
            self.dS = dS_v(degree=quad) + dS_h(degree=quad)
        else:
            self.ds = ds(degree=quad)
            self.dS = dS(degree=quad)

        # Add boundary conditions
        self.bcs = []
        if 'u' in eqns.bcs.keys():
            for bc in eqns.bcs['u']:
                self.bcs.append(
                    DirichletBC(self.fs.sub(0), bc.function_arg, bc.sub_domain)
                )

        # Set up test function and the vorticity term
        n = FacetNormal(domain.mesh)
        sign_u = 0.5*(sign(dot(u, n)) + 1)
        upw = lambda f: (sign_u('+')*f('+') + sign_u('-')*f('-'))

        if domain.mesh.topological_dimension == 2:
            mix_test = test_F - domain.perp(grad(test_Z))
            F_cross_u = Z*domain.perp(u)
        elif domain.mesh.topological_dimension == 3:
            mix_test = test_F - curl(test_Z)
            F_cross_u = cross(Z, u)

        time_deriv_form = inner(F, test_F)*dx + inner(Z, test_Z)*dx

        # Standard vector invariant transport form -----------------------------
        transport_form = (
            # vorticity term
            inner(mix_test, F_cross_u)*dx
            + inner(n, test_Z*Z*u)*self.ds
            # 0.5*grad(v . F)
            - 0.5 * div(mix_test) * inner(u, F)*dx
            + 0.5 * inner(mix_test, n) * inner(u, F)*self.ds
        )

        # Communtator of tranpose gradient terms -------------------------------
        # This is needed for general vector transport
        if transpose_commutator:
            u_dot_nabla_F = dot(u, transpose(nabla_grad(F)))
            transport_form += (
                - inner(n, test_Z*domain.perp(u_dot_nabla_F))*self.ds
                # + 0.5*grad(F).v
                - 0.5 * dot(F, div(outer(u, mix_test)))*dx
                + 0.5 * inner(mix_test('+'), n('+'))*dot(jump(u), upw(F))*self.dS
                # - 0.5*grad(v).F
                + 0.5 * dot(u, div(outer(F, mix_test)))*dx
                - 0.5 * inner(mix_test('+'), n('+'))*dot(jump(F), upw(u))*self.dS
            )

        # SUPG terms -----------------------------------------------------------
        # Add the vorticity residual to the transported vorticity,
        # which damps enstrophy
        if supg:

            # Determine SUPG coefficient ---------------------------------------
            tau = 0.5*domain.dt

            # Find mean grid spacing to determine a Courant number
            DG0 = FunctionSpace(domain.mesh, 'DG', 0)
            ones = Function(DG0).interpolate(Constant(1.0))
            area = assemble(ones*dx)
            mean_dx = (area/DG0.dof_count)**(1/domain.mesh.geometric_dimension)

            # Divide by approximately (1 + c)
            tau /= (1.0 + sqrt(dot(u, u))*domain.dt/Constant(mean_dx))

            dxqp = dx(degree=3)

            if domain.mesh.topological_dimension == 2:
                time_deriv_form -= inner(mix_test, tau*Z*domain.perp(u)/domain.dt)*dxqp
                transport_form -= inner(
                    mix_test, tau*domain.perp(u)*domain.divperp(Z*domain.perp(u))
                )*dxqp
                if transpose_commutator:
                    transport_form -= inner(
                        mix_test,
                        tau*domain.perp(u)*domain.divperp(u_dot_nabla_F)
                    )*dxqp
            elif domain.mesh.topological_dimension == 3:
                time_deriv_form -= inner(mix_test, tau*cross(Z, u)/domain.dt)*dxqp
                transport_form -= inner(
                    mix_test, tau*cross(curl(Z*u), u)
                )*dxqp
                if transpose_commutator:
                    transport_form -= inner(
                        mix_test,
                        tau*cross(curl(u_dot_nabla_F), u)
                    )*dxqp

        # Assemble the residual ------------------------------------------------
        residual = (
            time_derivative(time_deriv_form)
            + transport(
                transport_form, TransportEquationType.vector_invariant
            )
        )
        residual = transporting_velocity(residual, u)

        self.residual = subject(residual, self.X)

        self.x_in = Function(self.fs)
        self.Z_in = Function(V_vort)
        self.x_out = Function(self.fs)

        vort_test = TestFunction(V_vort)
        vort_trial = TrialFunction(V_vort)

        F_in, _ = split(self.x_in)

        eqn = (
            inner(vort_trial, vort_test)*dx
            + inner(domain.perp(grad(vort_test)), F_in)*dx
            + vort_test*inner(n, domain.perp(F_in))*self.ds
        )
        problem = LinearVariationalProblem(
            lhs(eqn), rhs(eqn), self.Z_in, constant_jacobian=True
        )
        self.solver = LinearVariationalSolver(problem)

    def pre_apply(self, x_in):
        """
        Sets the velocity field for the local mixed function.

        Args:
            x_in (:class:`Function`): The input velocity field
        """
        self.x_in.subfunctions[0].assign(x_in)

    def post_apply(self, x_out):
        """
        Sets the output velocity field from the local mixed function.

        Args:
            x_out (:class:`Function`): the output velocity field.
        """
        x_out.assign(self.x_out.subfunctions[0])

    def update(self, x_in_mixed):
        """
        Performs the solve to determine the vorticity function.

        Args:
            x_in_mixed (:class:`Function`): The mixed function to update.
        """
        self.x_in.subfunctions[0].assign(x_in_mixed.subfunctions[0])
        logger.debug('Vorticity solve')
        self.solver.solve()
        self.x_in.subfunctions[1].assign(self.Z_in)


class MeanMixingRatio(Augmentation):
    """
    This augments a transport problem involving a k=1 mixing ratio, by adding
    a mean mixing ratio field. This enables posivity to be
    ensured after each conservative transport step by blending the k=1 and 
    mean fields.

    Args:
        domain (:class:`Domain`): The domain object.
        eqns (:class:`PrognosticEquationSet`): The overarching equation set.
        mX_names (:class: list): A list of mixing ratios that
        require augmented mean mixing ratios.
    """

    def __init__(
            self, domain, eqns, mX_names
    ):

        self.name = 'mean_mixing_ratio'
        self.mX_names = mX_names
        self.mX_num = len(mX_names)
        self.orig_spaces = []

        # Store information about original equation set
        self.field_names = []
        for i in np.arange(len(eqns.field_names)):
            self.field_names.append(eqns.field_names[i])
            self.orig_spaces.append(eqns.spaces[i])

        self.eqn_orig = eqns
        self.domain = domain
        orig_spaces = eqns.spaces
        exist_spaces = eqns.spaces
        self.idx_orig = len(exist_spaces)

        DG0 = FunctionSpace(domain.mesh, "DG", 0)
        DG1 = FunctionSpace(domain.mesh, "DG", 1)

        # Set up fields and names for each mixing ratio
        self.mean_names = []
        self.mean_idxs = []
        self.mX_idxs = []
        mX_spaces = []
        mean_spaces = []
        self.rho_idxs = []

        for i in range(self.mX_num):
            mX_name = mX_names[i]
            self.mean_names.append('mean_'+mX_name)
            self.field_names.append(self.mean_names[-1])
            mean_spaces.append(DG0)
            exist_spaces.append(DG0)

            self.mean_idxs.append(self.idx_orig + i)

            # Extract the mixing ratio in question:
            mX_idx = eqns.field_names.index(mX_name)
            mX_spaces.append(eqns.spaces[mX_idx])
            self.mX_idxs.append(mX_idx)

            # Determine if this is a conservatively transported tracer.
            # If so, extract the corresponding density name, if not
            # set this to None.
            for tracer in eqns.active_tracers:
                if tracer.name == mX_name:
                    if tracer.density_name is not None:
                        self.rho_idxs.append(eqns.field_names.index(tracer.density_name))
                    else:
                        self.rho_idxs.append('None')

        # Define a limiter using the mean mixing ratios
        self.limiters = MeanLimiter(mX_spaces)

        # Projector for computing the mean mixing ratios
        self.DG1_field = Function(DG1) 
        self.rho_field = Function(DG1) 
        self.DG0_field = Function(DG0) 

        # Compute the mean mixing ratios using a conservative, consistent, projection.
        self.compute_mean_mX = ConservativeProjector(self.rho_field, self.rho_field, self.DG1_field, self.DG0_field, subtract_mean=True)

        # Create the new mixed function space
        #self.fs = MixedFunctionSpace(exist_spaces)

        #self.X = Function(self.fs)
        #self.tests = TestFunctions(self.fs)
        #self.x_in = Function(self.fs)
        #self.x_out = Function(self.fs)

        self.bcs = None


        # Leave the residual unmodified, as we update
        # the mean mixing ratios separately.
        # Also, leave the function spaces as the same
        self.residual = eqns.residual
        self.X = eqns.X
        self.tests = eqns.tests
        self.fs = MixedFunctionSpace(self.orig_spaces)
        self.x_in = Function(self.fs)
        self.x_out = Function(self.fs)

        # Make mean fields
        mean_fs = MixedFunctionSpace(mean_spaces)
        self.mean_fields = Function(mean_fs)

    def setup_residual(self, equation):
        """
        Copy the residual to the augmentation

        Args:
            equation (:class:`PrognosticEquationSet`): The overarching equation set.
            Note, this does not include the mean mixing ratios.
        """

        # Copy the existing residual
        self.residual = equation.residual

    def pre_apply(self, x_in):
        """
        Sets the original fields, i.e. not the mean fields

        Args:
            x_in (:class:`Function`): The input fields
        """

        for idx in range(self.idx_orig):
            self.x_in.subfunctions[idx].assign(x_in.subfunctions[idx])

    def post_apply(self, x_out):
        """
        Sets the output fields, i.e. not the mean fields

        Args:
            x_out (:class:`Function`): The output fields
        """
        for idx in range(self.idx_orig):
            x_out.subfunctions[idx].assign(self.x_out.subfunctions[idx])

    def update(self, x_in_mixed):
        """
        Compute the mean mixing ratio fields by conservative projection,
        where both the target and source density are in the higher-order
        space.

        Args:
            x_in_mixed (:class:`Function`): The mixed function, containing
            mean fields to update.
        """

        pass

    def limit(self, x_in_mixed):
        """
        Limit k=1 mixing ratios using a limiter that blends the
        k=1 field and its mean field.

        Args:
            x_in_mixed (:class:`Function`): The mixed function, containing
            mixing ratio fields to limit.
        """

        # Ensure non-negativity by applying the blended limiter
        mX_pre = []
        means = []

        # Compute the new mean mixing ratio
        for i in range(self.mX_num):
            self.rho_field.assign(x_in_mixed.subfunctions[self.rho_idxs[i]])

            # Compute the mean mixing ratio with conservative projection
            self.DG1_field.assign(x_in_mixed.subfunctions[self.mX_idxs[i]])
            self.compute_mean_mX.project()

            self.mean_fields.subfunctions[i].assign(self.DG0_field)

            mX_pre.append(x_in_mixed.subfunctions[self.mX_idxs[i]])
            means.append(self.mean_fields.subfunctions[i])

        self.limiters.apply(mX_pre, means)

        # Update the mixing ratios with the limited version
        for i in range(self.mX_num):
            x_in_mixed.subfunctions[self.mX_idxs[i]].assign(mX_pre[i])

            

        #for i in range(self.mX_num):
        #    mX_pre.append(x_in_mixed.subfunctions[self.mX_idxs[i]])
        #    means.append(x_in_mixed.subfunctions[self.mean_idxs[i]])

        #self.limiters.apply(mX_pre, means)

        #for i in range(self.mX_num):
            # SHouldn't need to do any clipping either ...
            #self.limiters._clip_DG1_field.apply(mX_pre[i], mX_pre[i])
        #    x_in_mixed.subfunctions[self.mX_idxs[i]].assign(mX_pre[i])
