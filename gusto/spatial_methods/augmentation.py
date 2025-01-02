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
    subject, all_terms, replace_subject, keep, replace_test_function,
    replace_trial_function, drop
)
from gusto import (
    time_derivative, transport, transporting_velocity, TransportEquationType,
    logger, prognostic, mass_weighted
)


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

        V_vel = domain.spaces('HDiv')
        V_vort = domain.spaces('H1')

        self.fs = MixedFunctionSpace((V_vel, V_vort))
        self.X = Function(self.fs)
        self.tests = TestFunctions(self.fs)

        u = Function(V_vel)
        F, Z = split(self.X)
        test_F, test_Z = self.tests

        if hasattr(domain.mesh, "_base_mesh"):
            self.ds = ds_b + ds_t + ds_v
            self.dS = dS_v + dS_h
        else:
            self.ds = ds
            self.dS = dS

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

        if domain.mesh.topological_dimension() == 2:
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
            mean_dx = (area/DG0.dof_count)**(1/domain.mesh.geometric_dimension())

            # Divide by approximately (1 + c)
            tau /= (1.0 + sqrt(dot(u, u))*domain.dt/Constant(mean_dx))

            dxqp = dx(degree=3)

            if domain.mesh.topological_dimension() == 2:
                time_deriv_form -= inner(mix_test, tau*Z*domain.perp(u)/domain.dt)*dxqp
                transport_form -= inner(
                    mix_test, tau*domain.perp(u)*domain.divperp(Z*domain.perp(u))
                )*dxqp
                if transpose_commutator:
                    transport_form -= inner(
                        mix_test,
                        tau*domain.perp(u)*domain.divperp(u_dot_nabla_F)
                    )*dxqp
            elif domain.mesh.topological_dimension() == 3:
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
    This augments a transport problem involving a mixing ratio to
    include a mean mixing ratio field. This enables posivity to be 
    ensured during conservative transport.

    Args:
        domain (:class:`Domain`): The domain object.
        eqns (:class:`PrognosticEquationSet`): The overarching equation set.
        mixing_ratio (:class: list): List of mixing ratios that 
        are to have augmented mean mixing ratio fields.
        OR, keep as a single mixing ratio, but define 
        multiple augmentations?
        START with just a single.
    """

    def __init__(
            self, domain, eqns, mX_name
    ):

        self.eqns = eqns
        exist_spaces = eqns.spaces

        self.idx_orig = len(exist_spaces)
        mean_idx = self.idx_orig

        # Extract the mixing ratio in question:
        mX_idx = eqns.field_names.index(mX_name)

        # Define the mean mixing ratio on the DG0 space
        DG0 = FunctionSpace(domain.mesh, "DG", 0)

         # Set up the scheme for the mean mixing ratio

        mean_mX = Function(DG0, name='mean_mX')
        mean_space = DG0
        exist_spaces.append(mean_space)

        self.fs = MixedFunctionSpace(exist_spaces)
        self.X = Function(self.fs)
        self.tests = TestFunctions(self.fs)

        print(self.X)

        self.bcs = []

        self.x_in = Function(self.fs)
        self.x_out = Function(self.fs)

        # Compute the new mean mass weighted term,
        # IF this is conservatively transported.
        mX_idx = eqns.field_names.index(mX_name)

        old_residual = eqns.residual

        mean_residual = old_residual.label_map(
            lambda t: t.get(prognostic) == mX_name,
            map_if_false=drop
        )

        # Replace trial functions with those in the new mixed function space
        for term in eqns.residual:
            print('\n')
            print(term.form)

        for idx in range(self.idx_orig):
            field = eqns.field_names[idx]
            # Seperate logic if mass-weighted or not?
            print(idx)
            print(field)

            prog = split(self.X)[idx]

            print('\n residual before change')
            print(old_residual.form)
            old_residual = old_residual.label_map(
                lambda t: t.get(prognostic) == field and not t.has_label(mass_weighted),
                map_if_true=replace_subject(self.X, old_idx=idx, new_idx = idx)
            )
            old_residual = old_residual.label_map(
                lambda t: t.get(prognostic) == field and not t.has_label(mass_weighted),
                map_if_true=replace_test_function(self.tests, old_idx=idx, new_idx=idx)
            )
            print('\n residual after change')
            print(old_residual.form)

            #old_residual = old_residual.label_map(
            #    lambda t: t.get(prognostic) == field and not t.has_label(mass_weighted),
            #    map_if_true=replace_subject(self.X[idx], old_idx=idx)
            #)
            #old_residual = old_residual.label_map(
            #    lambda t: t.get(prognostic) == field and not t.has_label(mass_weighted),
            #    map_if_true=replace_test_function(self.tests[idx], old_idx=idx)
            #)
    
        # Define the mean mixing ratio residual
        #mean_residual = mX_residual.label_map(
        #    lambda t: t.has_label(mass_weighted),
            #map_if_true=replace_subject(mean_mass, old_idx=mX_idx),
        #    map_if_false=replace_subject(self.X[mean_idx], old_idx = mX_idx)
        #)

        print('\n mean mX residual before change')
        print(mean_residual.form)
        mean_residual = mean_residual.label_map(
            all_terms,
            replace_subject(self.X, old_idx=mX_idx, new_idx = mean_idx)
        )
        mean_residual = mean_residual.label_map(
            all_terms,
            replace_test_function(self.tests, old_idx=mX_idx, new_idx = mean_idx)
        )
        print('\n mean mX residual after change')
        print(mean_residual.form)

        # Form the new residual
        self.residual = old_residual + mean_residual

        print('\n full residual')
        print(self.residual.form)

        

    def setup_residual(self):
        # Update the residual
        # Now, extract the existing residual:
        old_residual = eqns.residual

        print(old_residual.form)

        # Extract terms relating to the mixing ratio of interest
        mX_residual = old_residual.label_map(
            lambda t: t.get(prognostic) == mX_name,
            map_if_false=drop
        )

        print(mX_residual.form)

        # Replace trial and test functions with the new mixed
        # function space
        # Does this work is the subject is mass-weighted???
        for idx in range(self.idx_orig):
            field = eqns.field_names[idx]
            # Seperate logic if mass-weighted or not?
            print(idx)
            print(field)

            prog = split(self.X)[idx]

            #old_residual = old_residual.label_map(
            #    lambda t: t.get(prognostic) == field and not t.has_label(mass_weighted),
            #    map_if_true=replace_subject(self.X[idx], old_idx=idx)
            #)
            #old_residual = old_residual.label_map(
            #    lambda t: t.get(prognostic) == field and not t.has_label(mass_weighted),
            #    map_if_true=replace_test_function(self.tests[idx], old_idx=idx)
            #)
    
        # Define the mean mixing ratio residual
        mean_residual = mX_residual.label_map(
            lambda t: t.has_label(mass_weighted),
            #map_if_true=replace_subject(mean_mass, old_idx=mX_idx),
            map_if_false=replace_subject(self.X[mean_idx], old_idx = mX_idx)
        )

        mean_residual = mean_residual.label_map(
            lambda t: t.has_label(mass_weighted),
            map_if_false=replace_test_function(self.tests[mean_idx], old_idx=mX_idx)
        )

        # Form the new residual
        self.residual = old_residual + mean_residual


    def pre_apply(self, x_in):
        """
        Sets the original fields, i.e. not the mean mixing ratios

        Args:
            x_in (:class:`Function`): The input fields
        """

        #for idx, field in enumerate(self.eqn.field_names):
        #    self.x_in.subfunctions[idx].assign(x_in.subfunctions[idx])
        for idx in range(self.idx_orig):
            self.x_in.subfunctions[idx].assign(x_in.subfunctions[idx])


    def post_apply(self, x_out):
        """
        Sets the output fields, i.e. not the mean mixing ratios

        Args:
            x_out (:class:`Function`): The output fields
        """

        for idx, field in enumerate(self.eqn.field_names):
            x_out.subfunctions[idx].assign(self.x_out.subfunctions[idx])

    def update(self, x_in_mixed):
        """
        ...

        Args:
            x_in_mixed (:class:`Function`): The mixed function to update.
        """
        
        pass