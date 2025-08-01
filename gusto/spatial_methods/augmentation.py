"""
A module defining objects for temporarily augmenting an equation with another.
"""


from abc import ABCMeta, abstractmethod
from firedrake import (
    MixedFunctionSpace, Function, TestFunctions, split, inner, dx, grad,
    LinearVariationalProblem, LinearVariationalSolver, lhs, rhs, dot,
    ds_b, ds_v, ds_t, ds, FacetNormal, TestFunction, TrialFunction,
    transpose, nabla_grad, outer, dS, dS_h, dS_v, sign, jump, div,
    Constant, sqrt, cross, curl, FunctionSpace, assemble, DirichletBC,
    Projector, FiniteElement, TensorProductElement
)
from firedrake.fml import (
    subject, all_terms, replace_subject, replace_test_function, replace_trial_function,
    drop, Term, LabelledForm
)
from gusto import (
    time_derivative, transport, transporting_velocity, TransportEquationType,
    logger, prognostic, mass_weighted
)
from gusto.spatial_methods.limiters import MeanLimiter
from gusto.core.kernels import MeanValue
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

        if hasattr(domain.mesh, "_base_mesh"):
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
        mX_names (:class: list): A list of mixing ratios that
        require augmented mean mixing ratio fields.
    """

    def __init__(
            self, domain, eqns, mX_names
    ):

        self.name = 'mean_mixing_ratio'
        self.mX_names = mX_names
        self.mX_num = len(mX_names)

        # Store information about original equation set
        self.field_names = []
        for i in np.arange(len(eqns.field_names)):
            self.field_names.append(eqns.field_names[i])

        self.eqn_orig = eqns
        self.domain = domain
        exist_spaces = eqns.spaces
        self.idx_orig = len(exist_spaces)

        # Define the mean mixing ratio(s) in the DG0 space
        DG0 = FunctionSpace(domain.mesh, "DG", 0)
        DG1 = FunctionSpace(domain.mesh, "DG", 1)
        DG1_equispaced = domain.spaces('DG1_equispaced')

        # Set up fields and names for each mixing ratio
        self.mean_names = []
        self.mean_idxs = []
        self.mX_idxs = []
        mX_spaces = []
        mean_spaces = []
        self.limiters = []
        self.rho_names = []
        self.rho_idxs = []
        
        self.rho_name = self.field_names[0]

        # Define a lowest order density field
        #self.mean_rho_idx = self.idx_orig
        #exist_spaces.append(DG0)
        #self.field_names.append('mean_rho')
        #print(self.mean_rho_idx)

        for i in range(self.mX_num):
            mX_name = mX_names[i]
            self.mean_names.append('mean_'+mX_name)
            self.field_names.append(self.mean_names[-1])
            mean_spaces.append(DG0)
            exist_spaces.append(DG0)

            # Without rho0
            self.mean_idxs.append(self.idx_orig + i)

            # With rho0
            #self.mean_idxs.append(self.idx_orig + i + 1)

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
                        # With rho0
                        #self.rho_idxs.append(self.mean_rho_idx)
                    else:
                        self.rho_idxs.append('None')

        self.limiters = MeanLimiter(mX_spaces)

        # Contruct projectors for computing the mean mixing ratios
        #self.mX_ins = [Function(mX_spaces[i]) for i in range(self.mX_num)]
        #self.mean_outs = [Function(mean_spaces[i]) for i in range(self.mX_num)]
        #self.compute_means = [Projector(self.mX_ins[i], self.mean_outs[i])
        #                      for i in range(self.mX_num)]

        #self.mX_ins = [Function(DG0) for i in range(self.mX_num + 1)]
        #self.mean_outs = [Function(DG0) for i in range(self.mX_num + 1)]
        #self.compute_means = [Projector(self.mX_ins[i], self.mean_outs[i])
        #                      for i in range(self.mX_num + 1)]

        self.DG1_field = Function(DG1)
        self.rho_field = Function(DG1)
        self.rho_mean_field = Function(DG0)
        self.rho0_field = Function(DG0)
        self.DG0_field = Function(DG0)
        self.mean_evaluator = MeanValue(DG1)

       # self.compute_mean_rho = Projector(self.rho_field, self.rho0_field)

        # Add the subtract mean component in the future.
        #self.compute_mean_mX = ConservativeProjector(self.rho_field, self.rho0_field, self.DG1_field, self.DG0_field, subtract_mean=True)
        #self.compute_mean_mX = ConservativeProjector(self.rho_field, self.rho0_field, self.DG1_field, self.DG0_field)

        self.compute_mean_mX = ConservativeProjector(self.rho_field, self.rho_field, self.DG1_field, self.DG0_field)

        #self.compute_mean_mX = Projector(self.DG1_field, self.DG0_field)


        #self.DG1_field = Function(domain.spaces('DG1_equispaced'))
        #self.DG0_field = Function(FunctionSpace(domain.mesh, "DG", 0))
        #self.mean_evaluator = MeanValue(domain.spaces('DG1_equispaced'))

        #cell = domain.mesh._base_mesh.ufl_cell().cellname()
        #DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        #DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        #DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        #DG1_equispaced = FunctionSpace(domain.mesh, DG1_element)

        #DG1_equispaced = domain.spaces('DG1')

        #self.mX_ins = [Function(DG1_equispaced) for i in range(self.mX_num)]
        #self.mean_outs = [Function(mean_spaces[i]) for i in range(self.mX_num)]
        #self.compute_means = [Projector(self.mX_ins[i], self.mean_outs[i])
        #                      for i in range(self.mX_num)]
        

        # Create the new mixed function space:
        self.fs = MixedFunctionSpace(exist_spaces)
        self.X = Function(self.fs)
        self.tests = TestFunctions(self.fs)

        # New, keep the original function spaces the same
        #self.fs = eqns.function_space
        #self.X = eqns.X
        #self.tests = eqns.tests

        # Create the mean residual functions space
        #self.mean_fs = MixedFunctionSpace([DG0, DG0, DG0])
        #self.mean_X = Function(self.mean_fs)
        #self.mean_tests = TestFunctions(self.mean_fs)

        self.bcs = None

        self.x_in = Function(self.fs)
        self.x_out = Function(self.fs)

    def setup_residual(self, equation):

        # Copy the existing residual
        orig_residual = equation.residual

        print(len(orig_residual))

        # Replace tests and trials of original residual with
        # those from the new mixed space.
        # The indices of the original fields
        # will be the same in the new mixed space
        for idx in range(self.idx_orig):
            orig_residual = orig_residual.label_map(
                all_terms,
                replace_subject(self.X, old_idx=idx, new_idx=idx)
            )
            orig_residual = orig_residual.label_map(
                all_terms,
                replace_test_function(self.tests, old_idx=idx, new_idx=idx)
            )

        # Create the new residual
        new_residual = orig_residual

        print(len(new_residual))

        # CODE for when using a lowest order density field

        # Now, do the same for the mean density:
        #mean_rho_residual = equation.residual.label_map(
        #        lambda t: t.get(prognostic) == self.rho_name,
        #        map_if_false=drop
        #    )
        
        #print(len(mean_rho_residual))

        #mean_rho_residual = mean_rho_residual.label_map(
        #    all_terms,
        #    replace_subject(self.X, old_idx=0, new_idx=self.mean_rho_idx)
        #)

        # Replace test function from the new mixed space
        #mean_rho_residual = mean_rho_residual.label_map(
        #    all_terms,
        #    replace_test_function(self.tests, old_idx=0, new_idx=self.mean_rho_idx)
        # )

        # Update the name of the prognostic:
       # mean_rho_residual = mean_rho_residual.label_map(
         #   all_terms,
         #   lambda t: prognostic.update_value(t, 'mean_rho')
        #)

        #new_residual += mean_rho_residual

        #print(len(new_residual))

        # For each mean mixing ratio, copy across the terms relating to
        # the mixing ratio and replace the test function and trial function.
        for i in range(self.mX_num):
            mean_residual = equation.residual.label_map(
                lambda t: t.get(prognostic) == self.mX_names[i],
                map_if_false=drop
            )

            # Replace all instances of the original mixing ratio with
            # the mean version:
            for j in range(self.mX_num):
                mean_residual = mean_residual.label_map(
                    all_terms,
                    replace_subject(self.X, old_idx=self.mX_idxs[j], new_idx=self.mean_idxs[j])
                )

            # Replace test function from the new mixed space
            mean_residual = mean_residual.label_map(
                all_terms,
                replace_test_function(self.tests, old_idx=self.mX_idxs[i], new_idx=self.mean_idxs[i])
            )

            # Update the name of the prognostic:
            mean_residual = mean_residual.label_map(
                all_terms,
                lambda t: prognostic.update_value(t, self.mean_names[i])
            )

            # Append to the new residual
            new_residual += mean_residual

        self.residual = subject(new_residual, self.X)

        # Check these two forms
        print('\n Original equation with residual of length, ', len(equation.residual))
        print('\n Augmented equation with residual of length, ', len(self.residual))
        print('\n')


        # Check for any mass_weighted terms, and if so
        # replace the subject and test functions for 
        # the mass-weighted form.
        for term in self.residual:
            if term.has_label(mass_weighted):
                field = term.get(prognostic)
                mass_term = term.get(mass_weighted)

                # Transport terms are Terms not LabelledForms,
                # so this change this to use the label_map
                if term.has_label(transport):
                    old_mass_weighted_labels = mass_term.labels
                    mass_term = LabelledForm(mass_term)
                else:
                    old_mass_weighted_labels = mass_term.terms[0].labels

                if field in self.mX_names:
                    # Replace using mX_indices
                    list_idx = self.mX_names.index(field)
                    mX_idx = self.mX_idxs[list_idx]
                    rho_idx = self.rho_idxs[list_idx]

                    # Replace mixing ratio
                    mass_term_new = mass_term.label_map(
                        all_terms,
                        replace_subject(self.X, old_idx=mX_idx, new_idx=mX_idx)
                    )

                    # Replace original density
                    mass_term_new = mass_term_new.label_map(
                        all_terms,
                        replace_subject(self.X, old_idx=0, new_idx=0)
                    )

                    # Replace test function
                    mass_term_new = mass_term_new.label_map(
                        all_terms,
                        replace_test_function(self.tests, old_idx=mX_idx, new_idx=mX_idx)
                    )

                elif field in self.mean_names:
                    list_idx = self.mean_names.index(field)
                    mX_idx = self.mX_idxs[list_idx]
                    mean_idx = self.mean_idxs[list_idx]
                    rho_idx = self.rho_idxs[list_idx]

                    # Replace mixing ratio
                    mass_term_new = mass_term.label_map(
                        all_terms,
                        replace_subject(self.X, old_idx=mX_idx, new_idx=mean_idx)
                    )

                    # Replace density
                    mass_term_new = mass_term_new.label_map(
                        all_terms,
                        replace_subject(self.X, old_idx=0, new_idx=0)
                    )

                    # Replace test function
                    mass_term_new = mass_term_new.label_map(
                        all_terms,
                        replace_test_function(self.tests, old_idx=mX_idx, new_idx=mean_idx)
                    )

                # Create a new mass-weighted term, which has the correct labels
                mass_term_new = Term(mass_term_new.form, old_mass_weighted_labels)
                mass_term_new = subject(mass_term_new, self.X)

                # Make a new term, that links to the new mass-weighted term
                new_term = Term(term.form, term.labels)
                new_term = mass_weighted.update_value(new_term, mass_term_new)

                # Put this new term back in the residual:
                self.residual = self.residual.label_map(
                    lambda t: t == term,
                    map_if_true=lambda t: new_term
            )

        self.residual = subject(self.residual, self.X)

        print('Original residual')
        print('\n')
        for term in equation.residual:
            print(term.get(subject))
            print(term.labels)
            if term.has_label(mass_weighted):
                print('advective form of a mass weighted term')
                print(term.form)
                print('the mass-weighted part')
                print(term.get(mass_weighted).form)
                print(term.get(mass_weighted))
                #print(term.get(mass_weighted).labels)
                print('\n')
            else:
                print('Not')
                print(term.form)
                print('\n')

        # For debugging, check that all the replacements have worked:
        print('\n')
        for term in self.residual:
            print(term.get(subject))
            print(term.labels)
            if term.has_label(mass_weighted):
                print('advective form of a mass weighted term')
                print(term.form)
                print('the mass-weighted part')
                print(term.get(mass_weighted).form)
                print(term.get(mass_weighted))
                yo = term.get(mass_weighted)
                #print(term.get(mass_weighted).labels)
                print('\n')
            else:
                print('Not')
                print(term.form)
                print('\n')

    def pre_apply(self, x_in):
        """
        Sets the original fields, i.e. not the mean fields

        Args:
            x_in (:class:`Function`): The input fields
        """

        for idx in range(self.idx_orig):
            self.x_in.subfunctions[idx].assign(x_in.subfunctions[idx])

        # Check total mass conservation:
        X_sum = assemble(self.x_in.subfunctions[0]*self.x_in.subfunctions[1]*dx)
        X2_sum = assemble(self.x_in.subfunctions[0]*self.x_in.subfunctions[2]*dx)
        XT_sum = X_sum + 2*X2_sum

        self.DG1_field.interpolate(Constant(4e-6))
        XT_analyt = assemble(self.x_in.subfunctions[0]*self.DG1_field*dx)
        XT_diff = abs(XT_sum - XT_analyt)/XT_analyt

        print('\n Checking total conservation from 4e-6')
        print(XT_diff)

        if XT_diff > 1e-10:
            import sys; sys.exit()

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
        Compute the mean mixing ratio field by projecting the mixing
        ratio from DG1 into DG0.

        Args:
            x_in_mixed (:class:`Function`): The mixed function to update.
        """

        # Update the density field for the mean mixing ratios
        self.rho_field.assign(x_in_mixed.subfunctions[0])
        
        #x_in_mixed.subfunctions[self.mean_rho_idx].assign(self.rho_field)

        #self.compute_mean_rho.project()
        #x_in_mixed.subfunctions[self.mean_rho_idx].assign(self.rho0_field)


        # Update the mean mixing ratios:
        for i in range(self.mX_num):
            # Compute the mean mixing ratio with conservative projection
            self.DG1_field.assign(x_in_mixed.subfunctions[self.mX_idxs[i]])
            self.compute_mean_mX.project()

            print(f'\n min of mX field is {np.min(self.DG1_field.dat.data)}')
            print(f'\n min of mean field is {np.min(self.DG0_field.dat.data)}')

            # Clip zero values if needed:
            self.limiters._clip_means_kernel.apply(self.DG0_field, self.DG0_field)

            print(f'\n min of mean field after clipping is {np.min(self.DG0_field.dat.data)}')

            x_in_mixed.subfunctions[self.mean_idxs[i]].assign(self.DG0_field)

            # Debugging
            DG1_sum = assemble(self.rho_field*self.DG1_field*dx)
            #DG0_sum = assemble(x_in_mixed.subfunctions[3]*self.DG0_field*dx)
            #DG0_sum = assemble(self.rho0_field*self.DG0_field*dx)
            DG0_sum = assemble(self.rho_field*self.DG0_field*dx)
            rel_proj_err = np.abs(DG1_sum - DG0_sum)/DG1_sum
            print('rho*DG1*dx is', DG1_sum)
            print('rho*DG0*dx is', DG0_sum)
            print('What is the projection error? \n', rel_proj_err)
            #if rel_proj_err > 1e-14:
            #    import sys; sys.exit()

        # Check total mass conservation:
        X_sum = assemble(x_in_mixed.subfunctions[0]*x_in_mixed.subfunctions[1]*dx)
        X2_sum = assemble(x_in_mixed.subfunctions[0]*x_in_mixed.subfunctions[2]*dx)
        XT_sum = X_sum + 2*X2_sum

        self.DG1_field.interpolate(Constant(4e-6))
        XT_analyt = assemble(x_in_mixed.subfunctions[0]*self.DG1_field*dx)
        XT_diff = abs(XT_sum - XT_analyt)/XT_analyt

        print('\n Checking total conservation in augmnetation update')
        print(XT_diff)

        if XT_diff > 1e-10:
            import sys; sys.exit()

    def limit(self, x_in_mixed):
        # Ensure non-negativity by applying the blended limiter
        mX_pre = []
        means = []
        old_vals = []

        self.rho_field.assign(x_in_mixed.subfunctions[0])
        #self.rho0_field.assign(x_in_mixed.subfunctions[3])

         # What is the difference in density fields:
        #print('Original density field')
        #print(assemble(x_in_mixed.subfunctions[0]*dx))
        #print('Mean density field')
        #print(assemble(x_in_mixed.subfunctions[3]*dx))

        print('\n In the augmentation limiter: values after transport')
        for i in range(self.mX_num):
            mX_pre.append(x_in_mixed.subfunctions[self.mX_idxs[i]])
            means.append(x_in_mixed.subfunctions[self.mean_idxs[i]])

            print(f'\n min of {self.mX_names[i]} field:')
            print(np.min(mX_pre[i].dat.data))
            print(f'\n min of {self.mean_names[i]} field:')
            print(np.min(means[i].dat.data))
            print(f'\n max of {self.mX_names[i]} field:')
            print(np.max(mX_pre[i].dat.data))
            print(f'\n max of {self.mean_names[i]} field:')
            print(np.max(means[i].dat.data))
            print('Total of m_X field:')
            print(assemble(mX_pre[i]*dx))
            print('Total of mean field:')
            print(assemble(means[i]*dx))
            old_vals.append(assemble(self.rho_field*mX_pre[i]*dx))

            # Check difference in fields:
            DG1_sum = assemble(self.rho_field*mX_pre[i]*dx)
            #DG0_sum = assemble(x_in_mixed.subfunctions[3]*means[i]*dx)
            DG0_sum = assemble(self.rho_field*means[i]*dx)
            rel_proj_err = np.abs(DG1_sum - DG0_sum)/DG1_sum
            #print('rho1*dx', assemble(self.rho_field*dx) )
            #print('rho0*dx', assemble(x_in_mixed.subfunctions[3]*dx) )
            #print('rho diff', assemble((self.rho_field - x_in_mixed.subfunctions[3])*dx))
            print('DG1*dx', assemble(mX_pre[i]*dx) )
            print('DG0*dx', assemble(means[i]*dx) )
            print('rho*DG1*dx is', DG1_sum)
            print('rho*DG0*dx is', DG0_sum)
            print('What is the mass difference between the DG1 and DG0 fields? ', rel_proj_err)
            #if rel_proj_err > 1e-14:
            #    import sys; sys.exit()

        print('\n Applying the blended limiter')

        #self.limiters.apply(mX_pre, means)
        self.limiters.apply(mX_pre, means, self.rho_field)

        self.rho_field.interpolate(self.rho_field)

        print('\n After applying blended limiter')

        for i in range(self.mX_num):
            self.limiters._clip_DG1_field.apply(mX_pre[i],mX_pre[i])
            x_in_mixed.subfunctions[self.mX_idxs[i]].assign(mX_pre[i])
            print(f'\n min of {self.mX_names[i]} field:')
            print(np.min(mX_pre[i].dat.data))
            print(f'\n min of {self.mean_names[i]} field:')
            print(np.min(means[i].dat.data))
            print(f'\n max of {self.mX_names[i]} field:')
            print(np.max(mX_pre[i].dat.data))
            print(f'\n max of {self.mean_names[i]} field:')
            print(np.max(means[i].dat.data))

            print('\n Comparing tracer densities after limiting')
            new_sum = assemble(self.rho_field*mX_pre[i]*dx)
            rel_lim_err = np.abs(new_sum - old_vals[i])/old_vals[i]
            print('rho*m_old*dx is', old_vals[i])
            print('rho*m_new*dx is', new_sum)
            print('What is the mass error after limiting? ', rel_lim_err)
            #if rel_lim_err > 1e-14:
            #    import sys; sys.exit()

        # Check total mass conservation:
        X_sum = assemble(x_in_mixed.subfunctions[0]*x_in_mixed.subfunctions[1]*dx)
        X2_sum = assemble(x_in_mixed.subfunctions[0]*x_in_mixed.subfunctions[2]*dx)
        XT_sum = X_sum + 2*X2_sum

        self.DG1_field.interpolate(Constant(4e-6))
        XT_analyt = assemble(x_in_mixed.subfunctions[0]*self.DG1_field*dx)
        XT_diff = abs(XT_sum - XT_analyt)/XT_analyt

        print('\n Checking total conservation after limiting')
        print(XT_diff)

        if XT_diff > 1e-10:
            import sys; sys.exit()

