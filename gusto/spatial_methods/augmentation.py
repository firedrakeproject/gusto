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
    Projector
)
from firedrake.fml import (
    subject, all_terms, replace_subject, keep, replace_test_function,
    replace_trial_function, drop, Term, LabelledForm
)
from gusto import (
    time_derivative, transport, transporting_velocity, TransportEquationType,
    logger, prognostic, mass_weighted, nonlinear_time_derivative
)
from gusto.spatial_methods.limiters import MeanLimiter, MixedFSLimiter
import copy
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

        # Define the mean mixing ratio on the DG0 space
        DG0 = FunctionSpace(domain.mesh, "DG", 0)

        # Set up fields and names for each mixing ratio
        self.mean_names = []
        self.mean_idxs = []
        self.mX_idxs = []
        mX_spaces = []
        mean_spaces = []
        self.limiters = []
        self.rho_names = []
        self.rho_idxs = []
        #self.sublimiters = {}

        for i in range(self.mX_num):
            mX_name = mX_names[i]
            print(mX_names)
            print(mX_name)
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
            # set this to none.
            #self.rho_names.append('None')
            for tracer in eqns.active_tracers:
                if tracer.name == mX_name:
                    print(tracer.density_name)
                    if tracer.density_name is not None:
                        self.rho_idxs.append(eqns.field_names.index(tracer.density_name))
                    else:
                        self.rho_idxs.append('None')
            print(self.rho_idxs)
            # Define a limiter
            #self.limiters.append(MeanLimiter(eqns.spaces[mX_idx]))
            #self.sublimiters.update({mX_name: MeanLimiter(eqns.spaces[mX_idx])})

        #self.limiter = MixedFSLimiter(self.eqn_orig, self.sublimiters)
        #self.limiter = MixedFSLimiter(sublimiters)
        self.limiters = MeanLimiter(mX_spaces)

        # Contruct projectors for computing the mean mixing ratios
        self.mX_ins = [Function(mX_spaces[i]) for i in range(self.mX_num)]
        self.mean_outs = [Function(mean_spaces[i]) for i in range(self.mX_num)]
        self.compute_means = [Projector(self.mX_ins[i], self.mean_outs[i]) \
                               for i in range(self.mX_num)]

        # Create the new mixed function space:
        self.fs = MixedFunctionSpace(exist_spaces)
        self.X = Function(self.fs)
        self.tests = TestFunctions(self.fs)

        self.bcs = None

        self.x_in = Function(self.fs)
        self.x_out = Function(self.fs)

        print(eqns.field_names)
        print(self.field_names)


    # New attempt:
    def setup_residual(self, equation):
        # Copy across the residual, add terms for the mean
        # mixing ratio fields, and modify the subjects
        #
        # Additionally, cope the spatial method for the mixing ratio onto the 
        # mean mixing ratio.
        print('Setting up augmented residual')

        print(equation.residual.form)

        # The residual for the original equations
        orig_residual = equation.residual

        # Replace tests and trials of original residual with
        # those from the new mixed space.
        # The indices of the original fields
        # will be the same in the new mixed space
        for idx in range(self.idx_orig):
            print(idx, idx)

            print(orig_residual.form)

            orig_residual = orig_residual.label_map(
                all_terms,
                replace_subject(self.X, old_idx=idx, new_idx=idx)
            )
            orig_residual = orig_residual.label_map(
                all_terms,
                replace_test_function(self.tests, old_idx=idx, new_idx=idx)
            )
            print('\n')
            print(orig_residual.form)

            # Now, need to use replace subject for any mass_weighted terms?

        new_residual = orig_residual

        # For each mean mixing ratio, copy across the terms relating to 
        # the mixing ratio and replace the test function and trial function.
        for i in range(self.mX_num):
            mean_residual = equation.residual.label_map(
                lambda t: t.get(prognostic) == self.mX_names[i],
                map_if_false=drop
            )
            print(len(mean_residual))
            print(self.mX_idxs[i], self.mean_idxs[i])
            print('\n')
            print(mean_residual.form)

            # Replace any instances of the original mixing ratio with 
            # its mean version:
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
            print('\n')
            print(mean_residual.form)

            # Update the name of the prognostic:
            mean_residual = mean_residual.label_map(
                all_terms,
                lambda t: prognostic.update_value(t, self.mean_names[i])
            )

            new_residual += mean_residual

        self.residual = subject(new_residual, self.X)

        #Check these two forms
        print('\n Original equation with residual of length, ', len(equation.residual))
        print('\n Augmented equation with residual of length, ', len(self.residual))

        print('\n')
        #print(self.residual.form)

        # Yep, the mass_weighted terms still have the old form.
        print('\n')
        for term in self.residual:
            if term.has_label(mass_weighted):
                field = term.get(prognostic)
                print(field)
                print('advective form of a mass weighted term')
                #print(term)
                print(term.form)
                #print(term.labels)
                print('\n')
                print('the mass-weighted part')
                mass_term = term.get(mass_weighted)
                #print(mass_term)
                print(mass_term.form)
                print('\n')

                # Transport terms are Terms not LabelledForms,
                # so this needs to be changed. This will be revert later on.
                if term.has_label(transport):
                    mass_term = LabelledForm(mass_term)
                    #print(mass_term.form)
                    #print(mass_term.labels)
                    #mass_term = self.residual.label_map(
                    #    lambda t: t.has_label(transport) and t.get(prognostic) == field,
                    #    map_if_true=keep, map_if_false=drop
                    #)
                #    mass_term = self.residual.label_map(
                #    lambda t: t == term,
                #    map_if_false = drop
                #    )

                #    print(term.form)
                    
                #    print(term.form.terms)
                #    print('this is a transport term')
                 #   mass_term = term.get(mass_weighted).term
                #print('\n')
                    #print('extracting this term from the residual again')
                    #mass_term = mass_term.form.terms[0].get(mass_weighted)
                    #print(mass_term.form)
                    #print('\n')

                

                #mass_term.terms[0].replace_subject(self.X, 0, 0)
                #mass_term_new = replace_subject(mass_term.terms[0])(self.X, old_idx=0, new_idx=0)
                if field in self.mX_names:
                    # Replace using mX_indices
                    list_idx = self.mX_names.index(field)
                    mX_idx = self.mX_idxs[list_idx]
                    rho_idx = self.rho_idxs[list_idx]

                    # Replace mixing ratio
                    mass_term_new = mass_term.label_map(
                        all_terms,
                        replace_subject(self.X, old_idx = mX_idx, new_idx = mX_idx)
                    )

                    # Replace density
                    mass_term_new = mass_term_new.label_map(
                        all_terms,
                        replace_subject(self.X, old_idx = rho_idx, new_idx = rho_idx)
                    )

                    # Replace test function
                    mass_term_new = mass_term_new.label_map(
                        all_terms,
                        replace_test_function(self.tests, old_idx = mX_idx, new_idx = mX_idx)
                    )

                elif field in self.mean_names:
                    list_idx = self.mean_names.index(field)
                    mX_idx = self.mX_idxs[list_idx]
                    mean_idx = self.mean_idxs[list_idx]
                    rho_idx = self.rho_idxs[list_idx]

                    # Replace mixing ratio
                    mass_term_new = mass_term.label_map(
                        all_terms,
                        replace_subject(self.X, old_idx = mX_idx, new_idx = mean_idx)
                    )

                    # Replace density
                    mass_term_new = mass_term_new.label_map(
                        all_terms,
                        replace_subject(self.X, old_idx = rho_idx, new_idx = rho_idx)
                    )

                    # Replace test function
                    mass_term_new = mass_term_new.label_map(
                        all_terms,
                        replace_test_function(self.tests, old_idx = mX_idx, new_idx = mean_idx)
                    )

                print('form after changes')
                print(mass_term_new)
                print(mass_term_new.form)
                print('\n')

                # Carry across original labels here.
                mass_term_new = Term(mass_term_new.form, term.labels)

                #new_mass_weighted_term = Term(mass_term_new.form, term.labels)
                # Update the mass_weighted part of the residual term,
                # but be careful to get this correct for transport terms.
                new_term = Term(term.form, term.labels)
                new_term = mass_weighted.update_value(new_term, mass_term_new)

                #print('what are the types of these terms?')
                print(term)
                print(term.get(mass_weighted))
                print(new_term)
                print(mass_term_new)

                # Put this new term back in the residual:
                self.residual = self.residual.label_map(
                    lambda t: t == term,
                    map_if_true=lambda t: new_term
                )

                print('now we have hopefully replaced stuff')
                print('\n')
            else:
                print('Not')
                print(term.form)
                print('\n')

        print('\n Original equation with residual of length, ', len(equation.residual))
        print('\n Augmented equation with residual of length, ', len(self.residual))


        print('\n')
        for term in self.residual:
            print(term.get(subject))
            if term.has_label(mass_weighted):
                print('advective form of a mass weighted term')
                print(term.form)
                print('the mass-weighted part')
                print(term.get(mass_weighted).form)
                print(term.get(mass_weighted))
                print('\n')
            else:
                print('Not')
                print(term.form)
                print('\n')


    def pre_apply(self, x_in):
        """
        Sets the original fields, i.e. not the mean mixing ratios

        Args:
            x_in (:class:`Function`): The input fields
        """

        for idx in range(self.idx_orig):
            print(np.min(x_in.subfunctions[idx].dat.data))
            print('\n')
            self.x_in.subfunctions[idx].assign(x_in.subfunctions[idx])

    def post_apply(self, x_out):
        """
        Apply the limiters.
        Sets the output fields, i.e. not the mean mixing ratios

        Args:
            x_out (:class:`Function`): The output fields
        """

        for idx in range(self.idx_orig):
            print(np.min(self.x_out.subfunctions[idx].dat.data))
            print('\n')
            x_out.subfunctions[idx].assign(self.x_out.subfunctions[idx])

    def update(self, x_in_mixed):
        """
        Compute the mean mixing ratio field by projecting the mixing 
        ratio from DG1 into DG0.

        To DO: Shouldn't this be a conservative projection??!!
        This requires density fields...

        Args:
            x_in_mixed (:class:`Function`): The mixed function to update.
        """

        for i in range(self.mX_num):
            self.mX_ins[i].assign(x_in_mixed.subfunctions[self.mX_idxs[i]])
            print('\n min of mX field:')
            print(np.min(self.mX_ins[i].dat.data))
            self.compute_means[i].project()
            self.x_in.subfunctions[self.mean_idxs[i]].assign(self.mean_outs[i])

    def limit(self, x_in_mixed):
        # Ensure non-negativity by applying the blended limiter
        mX_pre = []
        means = []
        print('Values after transport')
        for i in range(self.mX_num):
            mX_pre.append(x_in_mixed.subfunctions[self.mX_idxs[i]])
            means.append(x_in_mixed.subfunctions[self.mean_idxs[i]])

            print('\n min of mX field:')
            print(np.min(mX_pre[i].dat.data))
            print(f'\n min of {self.mean_names[i]} field:')
            print(np.min(means[i].dat.data))
            print('\n max of mX field:')
            print(np.max(mX_pre[i].dat.data))
            print(f'\n max of {self.mean_names[i]} field:')
            print(np.max(means[i].dat.data))

        self.limiters.apply(mX_pre, means)

        print('\n After applying blended limiter')

        for i in range(self.mX_num):
            x_in_mixed.subfunctions[self.mX_idxs[i]].assign(mX_pre[i])
            print('\n min of mX field:')
            print(np.min(mX_pre[i].dat.data))
            print(f'\n min of {self.mean_names[i]} field:')
            print(np.min(means[i].dat.data))
            print('\n max of mX field:')
            print(np.max(mX_pre[i].dat.data))
            print(f'\n max of {self.mean_names[i]} field:')
            print(np.max(means[i].dat.data))

