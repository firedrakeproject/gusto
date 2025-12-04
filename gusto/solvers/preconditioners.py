"""A module containing specialised preconditioners for Gusto applications."""

from firedrake import (dot, jump, dS_h, ds_b, ds_t, ds,
                       FacetNormal, Tensor, AssembledVector,
                       AuxiliaryOperatorPC, PETSc)

from firedrake.preconditioners import PCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from gusto.recovery.recovery_kernels import AverageKernel, AverageWeightings
from pyop2.profiling import timed_region, timed_function
from functools import partial
from numpy import arange


__all__ = ["VerticalHybridizationPC", "SlateSchurPC", "AuxiliaryPC", "CompressibleHybridisedSCPC"]


class AuxiliaryPC(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        a, bcs = self.get_appctx(pc)['auxform']
        return (a(test, trial), bcs)


class SlateSchurPC(AuxiliaryOperatorPC):
    _prefix = "slateschur_"

    def form(self, pc, test, trial):
        appctx = self.get_appctx(pc)

        aform = appctx['slateschur_form']
        Va = aform.arguments()[0].function_space()
        Vlen = len(Va)

        # which fields are in the schur complement?
        pc_prefix = pc.getOptionsPrefix() + "pc_" + self._prefix
        fields = PETSc.Options().getIntArray(
            pc_prefix + 'fields', [Vlen-1])
        nfields = len(fields)

        if nfields > Vlen - 1:
            raise ValueError("Must have at least one uneliminated field")

        first_fields = arange(nfields)
        last_fields = arange(Vlen - nfields, Vlen)

        # eliminate fields not in the schur complement
        eliminate_first = all(fields == last_fields)
        eliminate_last = all(fields == first_fields)

        if not any((eliminate_first, eliminate_last)):
            raise ValueError(
                "Can only eliminate contiguous fields at the"
                f" beginning {first_fields} or end {last_fields}"
                f" of function space, not {fields}")

        a = Tensor(aform)
        if eliminate_first:
            n = Vlen - nfields
            a00 = a.blocks[:n, :n]
            a10 = a.blocks[n:, :n]
            a01 = a.blocks[:n, n:]
            a11 = a.blocks[n:, n:]
        elif eliminate_last:
            n = nfields
            a00 = a.blocks[n:, n:]
            a10 = a.blocks[:n, n:]
            a01 = a.blocks[n:, :n]
            a11 = a.blocks[:n, :n]

        schur_complement = a11 - a10*a00.inv*a01

        return (schur_complement, None)

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if hasattr(self, "pc"):
            msg = "PC to approximate Schur complement using Slate.\n"
            viewer.printfASCII(msg)
            self.pc.view(viewer)


class VerticalHybridizationPC(PCBase):
    """
    A preconditioner for the vertical hydrostatic pressure system.

    A Slate-based python preconditioner for solving the hydrostatic pressure
    equation (after rewriting as a mixed vertical HDiv x L2 system). This
    preconditioner hybridizes a mixed system in the vertical direction. This
    means that velocities are rendered discontinuous in the vertical and
    Lagrange multipliers are introduced on the top/bottom facets to weakly
    enforce continuity through the top/bottom faces of each cell.

    This PC assembles a statically condensed problem for the multipliers and
    inverts the resulting system using the provided solver options. The original
    unknowns are recovered element-wise by solving local linear systems.

    All elimination and recovery kernels are generated using the Slate DSL in
    Firedrake. See firedrake/preconditioners/base.py for more details.
    """

    @timed_function("VertHybridInit")
    def initialize(self, pc):
        """
        Set up the problem context.

        Takes the original mixed problem and transforms it into the equivalent
        hybrid-mixed system. A KSP object is created for the Lagrange
        multipliers on the top/bottom faces of the mesh cells.

        Args:
            pc (:class:`PETSc.PC`): preconditioner object to initialize.
        """
        from firedrake import (FunctionSpace, Function, Constant, Cofunction,
                               FiniteElement, TensorProductElement,
                               TrialFunction, TrialFunctions, TestFunction,
                               DirichletBC, interval, MixedElement, BrokenElement)
        from firedrake.assemble import get_assembler
        from firedrake.formmanipulation import split_form
        from ufl.algorithms.replace import replace
        from ufl.cell import TensorProductCell

        # Extract PC context
        prefix = pc.getOptionsPrefix() + "vert_hybridization_"
        _, P = pc.getOperators()
        self.ctx = P.getPythonContext()

        if not isinstance(self.ctx, ImplicitMatrixContext):
            raise ValueError("The python context must be an ImplicitMatrixContext")

        test, trial = self.ctx.a.arguments()

        V = test.function_space()
        mesh = V.mesh()
        try:
            from firedrake import MeshSequenceGeometry  # noqa: F401

            unique_mesh = mesh.unique()
        except ImportError:
            unique_mesh = mesh

        # Magically determine which spaces are vector and scalar valued
        for i, Vi in enumerate(V):

            # Vector-valued spaces will have a non-empty value_shape
            if Vi.value_shape:
                self.vidx = i
            else:
                self.pidx = i

        Vv = V[self.vidx]
        Vp = V[self.pidx]

        # Create the space of approximate traces in the vertical.
        # NOTE: Technically a hack since the resulting space is technically
        # defined in cell interiors, however the degrees of freedom will only
        # be geometrically defined on edges. Arguments will only be used in
        # surface integrals
        deg, _ = Vv.ufl_element().degree()

        # Assumes a tensor product cell (quads, triangular-prisms, cubes)
        if not isinstance(Vp.ufl_element().cell, TensorProductCell):
            raise NotImplementedError("Currently only implemented for tensor product discretizations")

        # Only want the horizontal cell
        cell, _ = Vp.ufl_element().cell._cells

        DG = FiniteElement("DG", cell, deg)
        CG = FiniteElement("CG", interval, 1)
        Vv_tr_element = TensorProductElement(DG, CG)
        Vv_tr = FunctionSpace(unique_mesh, Vv_tr_element)

        # Break the spaces
        broken_elements = MixedElement([BrokenElement(Vi.ufl_element()) for Vi in V])
        V_d = FunctionSpace(mesh, broken_elements)

        # Set up relevant functions
        self.broken_solution = Function(V_d)
        self.broken_residual = Cofunction(V_d.dual())
        self.trace_solution = Function(Vv_tr)
        self.unbroken_solution = Function(V)
        self.unbroken_residual = Cofunction(V.dual())

        weight_kernel = AverageWeightings(Vv)
        self.weight = Function(Vv)
        weight_kernel.apply(self.weight)

        # Averaging kernel
        self.average_kernel = AverageKernel(Vv)

        # Original mixed operator replaced with "broken" arguments
        arg_map = {test: TestFunction(V_d),
                   trial: TrialFunction(V_d)}
        Atilde = Tensor(replace(self.ctx.a, arg_map))
        gammar = TestFunction(Vv_tr)
        n = FacetNormal(unique_mesh)
        sigma = TrialFunctions(V_d)[self.vidx]

        # Again, assumes tensor product structure. Why use this if you
        # don't have some form of vertical extrusion?
        Kform = gammar('+') * jump(sigma, n=n) * dS_h

        # Here we deal with boundary conditions
        if self.ctx.row_bcs:
            # Find all the subdomains with neumann BCS
            # These are Dirichlet BCs on the vidx space
            neumann_subdomains = set()
            for bc in self.ctx.row_bcs:
                if bc.function_space().index == self.pidx:
                    raise NotImplementedError("Dirichlet conditions for scalar variable not supported. Use a weak bc.")
                if bc.function_space().index != self.vidx:
                    raise NotImplementedError("Dirichlet bc set on unsupported space.")
                # append the set of sub domains
                subdom = bc.sub_domain
                if isinstance(subdom, str):
                    neumann_subdomains |= set([subdom])
                else:
                    neumann_subdomains |= set(subdom)

            # separate out the top and bottom bcs
            extruded_neumann_subdomains = neumann_subdomains & {"top", "bottom"}
            neumann_subdomains = neumann_subdomains - extruded_neumann_subdomains

            integrand = gammar * dot(sigma, n)
            measures = []
            trace_subdomains = []
            for subdomain in sorted(extruded_neumann_subdomains):
                measures.append({"top": ds_t, "bottom": ds_b}[subdomain])
                trace_subdomains.extend(sorted({"top", "bottom"} - extruded_neumann_subdomains))

            measures.extend((ds(sd) for sd in sorted(neumann_subdomains)))
            markers = [int(x) for x in unique_mesh.exterior_facets.unique_markers]
            dirichlet_subdomains = set(markers) - neumann_subdomains
            trace_subdomains.extend(sorted(dirichlet_subdomains))

            for measure in measures:
                Kform += integrand * measure

        else:
            trace_subdomains = ["top", "bottom"]

        trace_bcs = [DirichletBC(Vv_tr, Constant(0.0), subdomain)
                     for subdomain in trace_subdomains]

        # Make a SLATE tensor from Kform
        K = Tensor(Kform)

        # Assemble the Schur complement operator and right-hand side
        self.schur_rhs = Cofunction(Vv_tr.dual())
        self._assemble_Srhs = partial(get_assembler(
            K * Atilde.inv * AssembledVector(self.broken_residual),
            form_compiler_parameters=self.ctx.fc_params).assemble, tensor=self.schur_rhs)

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")

        schur_comp = K * Atilde.inv * K.T
        self.S = get_assembler(schur_comp, bcs=trace_bcs,
                               form_compiler_parameters=self.ctx.fc_params,
                               mat_type=mat_type,
                               options_prefix=prefix).allocate()
        self._assemble_S = partial(get_assembler(
            schur_comp,
            bcs=trace_bcs,
            form_compiler_parameters=self.ctx.fc_params).assemble, tensor=self.S)

        self._assemble_S()
        Smat = self.S.petscmat

        nullspace = self.ctx.appctx.get("vert_trace_nullspace", None)
        if nullspace is not None:
            nsp = nullspace(Vv_tr)
            Smat.setNullSpace(nsp.nullspace(comm=pc.comm))

        # Set up the KSP for the system of Lagrange multipliers
        trace_ksp = PETSc.KSP().create(comm=pc.comm)
        trace_ksp.setOptionsPrefix(prefix)
        trace_ksp.setOperators(Smat)
        trace_ksp.setUp()
        trace_ksp.setFromOptions()
        self.trace_ksp = trace_ksp

        split_mixed_op = dict(split_form(Atilde.form))
        split_trace_op = dict(split_form(K.form))

        # Generate reconstruction calls
        self._reconstruction_calls(split_mixed_op, split_trace_op)

    def _reconstruction_calls(self, split_mixed_op, split_trace_op):
        """
        Generates reconstruction calls.

        This generates the reconstruction calls for the unknowns using the
        Lagrange multipliers.

        Args:
            split_mixed_op (dict): a ``dict`` of split forms that make up the
                broken mixed operator from the original problem.
            split_trace_op (dict): a ``dict`` of split forms that make up the
                trace contribution in the hybridized mixed system.
        """
        from firedrake.assemble import get_assembler

        # We always eliminate the velocity block first
        id0, id1 = (self.vidx, self.pidx)

        # TODO: When PyOP2 is able to write into mixed dats,
        # the reconstruction expressions can simplify into
        # one clean expression.
        A = Tensor(split_mixed_op[(id0, id0)])
        B = Tensor(split_mixed_op[(id0, id1)])
        C = Tensor(split_mixed_op[(id1, id0)])
        D = Tensor(split_mixed_op[(id1, id1)])
        K_0 = Tensor(split_trace_op[(0, id0)])
        K_1 = Tensor(split_trace_op[(0, id1)])

        # Split functions and reconstruct each bit separately
        split_residual = self.broken_residual.subfunctions
        split_sol = self.broken_solution.subfunctions
        g = AssembledVector(split_residual[id0])
        f = AssembledVector(split_residual[id1])
        sigma = split_sol[id0]
        u = split_sol[id1]
        lambdar = AssembledVector(self.trace_solution)

        M = D - C * A.inv * B
        R = K_1.T - C * A.inv * K_0.T
        u_rec = M.solve(f - C * A.inv * g - R * lambdar,
                        decomposition="PartialPivLU")
        self._sub_unknown = partial(get_assembler(
            u_rec,
            form_compiler_parameters=self.ctx.fc_params).assemble, tensor=u)

        sigma_rec = A.solve(g - B * AssembledVector(u) - K_0.T * lambdar,
                            decomposition="PartialPivLU")
        self._elim_unknown = partial(get_assembler(
            sigma_rec,
            form_compiler_parameters=self.ctx.fc_params).assemble, tensor=sigma)

    @timed_function("VertHybridRecon")
    def _reconstruct(self):
        """
        Reconstructs the system unknowns using the multipliers.

        Note that the reconstruction calls are assumed to be
        initialized at this point.
        """

        # We assemble the unknown which is an expression
        # of the first eliminated variable.
        self._sub_unknown()
        # Recover the eliminated unknown
        self._elim_unknown()

    @timed_function("VertHybridUpdate")
    def update(self, pc):
        """
        Update by assembling into the operator.

        Args:
            pc (:class:`PETSc.PC`): preconditioner object.
        """
        # No need to reconstruct symbolic objects.
        self._assemble_S()

    def apply(self, pc, x, y):
        """
        Apply the preconditioner to x, putting the result in y.

        We solve the forward eliminated problem for the approximate traces of
        the scalar solution (the multipliers) and reconstruct the "broken flux
        and scalar variable."

        Lastly, we project the broken solutions into the mimetic non-broken
        finite element space.

        Args:
            pc (:class:`PETSc.PC`): the preconditioner object.
            x (:class:`PETSc.Vec`): the vector to apply the preconditioner to.
            y (:class:`PETSc.Vec`): the vector to put the result into.
        """

        with timed_region("VertHybridBreak"):
            with self.unbroken_residual.dat.vec_wo as v:
                x.copy(v)

            # Transfer unbroken_rhs into broken_rhs
            # NOTE: Scalar space is already "broken" so no need for
            # any projections
            unbroken_scalar_data = self.unbroken_residual.subfunctions[self.pidx]
            broken_scalar_data = self.broken_residual.subfunctions[self.pidx]
            unbroken_scalar_data.dat.copy(broken_scalar_data.dat)

        with timed_region("VertHybridRHS"):
            # Assemble the new "broken" hdiv residual
            # We need a residual R' in the broken space that
            # gives R'[w] = R[w] when w is in the unbroken space.
            # We do this by splitting the residual equally between
            # basis functions that add together to give unbroken
            # basis functions.

            unbroken_res_hdiv = self.unbroken_residual.subfunctions[self.vidx]
            broken_res_hdiv = self.broken_residual.subfunctions[self.vidx]
            broken_res_hdiv.assign(0)
            self.average_kernel.apply(broken_res_hdiv, self.weight, unbroken_res_hdiv)

            # Compute the rhs for the multiplier system
            self._assemble_Srhs()

        with timed_region("VertHybridSolve"):
            # Solve the system for the Lagrange multipliers
            with self.schur_rhs.dat.vec_ro as b:
                if self.trace_ksp.getInitialGuessNonzero():
                    acc = self.trace_solution.dat.vec
                else:
                    acc = self.trace_solution.dat.vec_wo
                with acc as x_trace:
                    self.trace_ksp.solve(b, x_trace)

        # Reconstruct the unknowns
        self._reconstruct()

        with timed_region("VertHybridRecover"):
            # Project the broken solution into non-broken spaces
            broken_pressure = self.broken_solution.subfunctions[self.pidx]
            unbroken_pressure = self.unbroken_solution.subfunctions[self.pidx]
            broken_pressure.dat.copy(unbroken_pressure.dat)

            # Compute the hdiv projection of the broken hdiv solution
            broken_hdiv = self.broken_solution.subfunctions[self.vidx]
            unbroken_hdiv = self.unbroken_solution.subfunctions[self.vidx]
            unbroken_hdiv.assign(0)

            self.average_kernel.apply(unbroken_hdiv, self.weight, broken_hdiv)

            with self.unbroken_solution.dat.vec_ro as v:
                v.copy(y)

    def applyTranspose(self, pc, x, y):
        """
        Apply the transpose of the preconditioner.

        Args:
            pc (:class:`PETSc.PC`): the preconditioner object.
            x (:class:`PETSc.Vec`): the vector to apply the preconditioner to.
            y (:class:`PETSc.Vec`): the vector to put the result into.

        Raises:
            NotImplementedError: this method is currently not implemented.
        """

        raise NotImplementedError("The transpose application of the PC is not implemented.")

    def view(self, pc, viewer=None):
        """
        Viewer calls for the various configurable objects in this PC.

        Args:
            pc (:class:`PETSc.PC`): the preconditioner object.
            viewer (:class:`PETSc.Viewer`, optional): viewer object. Defaults to
                None.
        """

        super(VerticalHybridizationPC, self).view(pc, viewer)
        viewer.pushASCIITab()
        viewer.printfASCII("Solves K * P^-1 * K.T using local eliminations.\n")
        viewer.printfASCII("KSP solver for the multipliers:\n")
        viewer.pushASCIITab()
        self.trace_ksp.view(viewer)
        viewer.popASCIITab()
        viewer.printfASCII("Locally reconstructing the broken solutions from the multipliers.\n")
        viewer.pushASCIITab()
        viewer.printfASCII("Project the broken hdiv solution into the HDiv space.\n")
        viewer.popASCIITab()

class CompressibleHybridisedSCPC(PCBase):
    """
    A bespoke hybridised preconditioner for the compressible Euler equations.

    This solves a linear problem for the compressible Euler equations in
    theta-exner formulation with prognostic variables u (velocity), rho
    (density) and theta (potential temperature). It follows the following
    strategy:

    (1) Analytically eliminate theta (introduces error near topography)

    (2a) Formulate the resulting mixed system for u and rho using a
         hybridized mixed method. This breaks continuity in the
         linear perturbations of u, and introduces a new unknown on the
         mesh interfaces approximating the average of the Exner pressure
         perturbations. These trace unknowns also act as Lagrange
         multipliers enforcing normal continuity of the "broken" u variable.

    (2b) Statically condense the block-sparse system into a single system
         for the Lagrange multipliers. This is the only globally coupled
         system requiring a linear solver.

    (2c) Using the computed trace variables, we locally recover the
         broken velocity and density perturbations. This is accomplished
         in two stages:
         (i): Recover rho locally using the multipliers.
         (ii): Recover "broken" u locally using rho and the multipliers.

    (2d) Project the "broken" velocity field into the HDiv-conforming
         space using local averaging.

    (3) Reconstruct theta
    """

    _prefix="compressible_hybrid_scpc"

    scpc_parameters = {'mat_type': 'matfree',
                        'ksp_type': 'preonly',
                        'ksp_converged_reason': None,
                        'ksp_monitor_true_residual': None,
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.SCPC',
                        'pc_sc_eliminate_fields': '0, 1',
                        # The reduced operator is not symmetric
                        'condensed_field': {'ksp_type': 'fgmres',
                                            'ksp_rtol': 1.0e-8,
                                            'ksp_atol': 1.0e-8,
                                            'ksp_max_it': 100,
                                            'pc_type': 'gamg',
                                            'pc_gamg_sym_graph': None,
                                            'mg_levels': {'ksp_type': 'gmres',
                                                        'ksp_max_it': 5,
                                                        'pc_type': 'bjacobi',
                                                        'sub_pc_type': 'ilu'}}}

    def initialize(self, pc):
        """
        Set up problem and solver
        """
        from firedrake import (split, LinearVariationalProblem, LinearVariationalSolver,
                                TestFunctions, TrialFunctions, TestFunction, TrialFunction, lhs,
                                rhs, FacetNormal, div, dx, jump, avg, dS_v, dS_h, ds_v, ds_t, ds_b,
                                ds_tb, inner, dot, grad, Function, cross,
                                BrokenElement, FunctionSpace, MixedFunctionSpace, as_vector,
                                Cofunction, Constant)
        from gusto.equations.active_tracers import TracerVariableType
        from gusto.core.labels import hydrostatic
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        self._process_context(pc)

        # Equations and parameters
        equations = self.equations
        dt = self.dt
        cp = equations.parameters.cp

        # Set relaxation parameters. If an alternative has not been given, set
        # to semi-implicit off-centering factor
        beta_u = dt*self.tau_values.get("u", self.alpha)
        beta_t = dt*self.tau_values.get("theta", self.alpha)
        beta_r = dt*self.tau_values.get("rho", self.alpha)

        # Get function spaces
        self.Vu = self.equations.domain.spaces("HDiv")
        self.Vrho = self.equations.domain.spaces("DG")
        self.Vtheta = self.equations.domain.spaces("theta")

        # Build trace and broken spaces
        self.Vu_broken = FunctionSpace(self.mesh, BrokenElement(self.Vu.ufl_element()))
        h_deg = self.Vrho.ufl_element().degree()[0]
        v_deg = self.Vrho.ufl_element().degree()[1]
        self.Vtrace = FunctionSpace(self.mesh, "HDiv Trace", degree=(h_deg, v_deg))

        # Mixed Function Spaces
        self.W = equations.function_space # (Vu, Vrho, Vtheta)
        self.W_hyb = MixedFunctionSpace((self.Vu_broken, self.Vrho, self.Vtrace))

        # Define
        self.xstar = Cofunction(self.W.dual())
        self.xrhs = Function(self.W)
        self.y = Function(self.W)

        self.y_hybrid = Function(self.W_hyb)

        u_in, rho_in, theta_in = self.xrhs.subfunctions[0:3]

        # Get bcs
        self.bcs = self.equations.bcs['u']

        # Set up hybridized solver for (u, rho, l) system
        w, phi, dl = TestFunctions(self.W_hyb)
        u, rho, l0 = TrialFunctions(self.W_hyb)
        n = FacetNormal(self.mesh)

        # Get background fields
        _, rhobar, thetabar = split(self.equations.X_ref)[0:3]
        kappa = equations.parameters.kappa
        R_d = equations.parameters.R_d
        p_0 = equations.parameters.p_0
        exnerbar =  (rhobar * R_d * thetabar / p_0) ** (kappa / (1 - kappa))
        exnerbar_rho = (kappa / (1 - kappa)) * (rhobar * R_d * thetabar / p_0) ** (kappa / (1 - kappa)) / rhobar
        exnerbar_theta = (kappa / (1 - kappa)) * (rhobar * R_d * thetabar / p_0) ** (kappa / (1 - kappa)) / thetabar

        # Analytical (approximate) elimination of theta
        k = equations.domain.k       # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta_t + theta_in

        # Only include theta' (rather than exner') in the vertical
        # component of the gradient

        # The exner prime term (here, bars are for mean and no bars are
        # for linear perturbations)
        exner = exnerbar_theta*theta + exnerbar_rho*rho

        # Vertical projection
        def V(u):
            return k*inner(u, k)

        # hydrostatic projection
        h_project = lambda u: u - k*inner(u, k)

        # Specify degree for some terms as estimated degree is too large
        dx_qp = dx(degree=(equations.domain.max_quad_degree))
        dS_v_qp = dS_v(degree=(equations.domain.max_quad_degree))
        dS_h_qp = dS_h(degree=(equations.domain.max_quad_degree))
        ds_v_qp = ds_v(degree=(equations.domain.max_quad_degree))
        ds_tb_qp = (ds_t(degree=(equations.domain.max_quad_degree))
                    + ds_b(degree=(equations.domain.max_quad_degree)))

        # Add effect of density of water upon theta, using moisture reference profiles
        # TODO: Explore if this is the right thing to do for the linear problem
        if equations.active_tracers is not None:
            mr_t = Constant(0.0)*thetabar
            for tracer in equations.active_tracers:
                if tracer.chemical == 'H2O':
                    if tracer.variable_type == TracerVariableType.mixing_ratio:
                        idx = equations.field_names.index(tracer.name)
                        mr_bar = split(equations.X_ref)[idx]
                        mr_t += mr_bar
                    else:
                        raise NotImplementedError('Only mixing ratio tracers are implemented')

            theta_w = theta / (1 + mr_t)
            thetabar_w = thetabar / (1 + mr_t)
        else:
            theta_w = theta
            thetabar_w = thetabar


        _l0 = TrialFunction(self.Vtrace)
        _dl = TestFunction(self.Vtrace)
        a_tr = _dl('+')*_l0('+')*(dS_v_qp + dS_h_qp) + _dl*_l0*ds_v_qp + _dl*_l0*ds_tb_qp

        def L_tr(f):
            return _dl('+')*avg(f)*(dS_v_qp + dS_h_qp) + _dl*f*ds_v_qp + _dl*f*ds_tb_qp

        cg_ilu_parameters = {'ksp_type': 'cg',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}

        # Project field averages into functions on the trace space
        rhobar_avg = Function(self.Vtrace)
        exnerbar_avg = Function(self.Vtrace)

        rho_avg_prb = LinearVariationalProblem(a_tr, L_tr(rhobar), rhobar_avg,
                                               constant_jacobian=True)
        exner_avg_prb = LinearVariationalProblem(a_tr, L_tr(exnerbar), exnerbar_avg,
                                                 constant_jacobian=True)

        self.rho_avg_solver = LinearVariationalSolver(rho_avg_prb,
                                                      solver_parameters=cg_ilu_parameters,
                                                      options_prefix=pc.getOptionsPrefix()+'rhobar_avg_solver')
        self.exner_avg_solver = LinearVariationalSolver(exner_avg_prb,
                                                        solver_parameters=cg_ilu_parameters,
                                                        options_prefix=pc.getOptionsPrefix()+'exnerbar_avg_solver')

        # # "broken" u, rho, and trace system
        # # NOTE: no ds_v integrals since equations are defined on
        # # a periodic (or sphere) base mesh.
        if  any([t.has_label(hydrostatic) for t in equations.residual]):
            u_mass = inner(w, (h_project(u) - u_in))*dx
        else:
            u_mass = inner(w, (u - u_in ))*dx

        eqn = (
            # momentum equation
            u_mass
            - beta_u*cp*div(theta_w*V(w))*exnerbar*dx_qp
            # following does nothing but is preserved in the comments
            # to remind us why (because V(w) is purely vertical).
            # + beta*cp*jump(theta_w*V(w), n=n)*exnerbar_avg('+')*dS_v_qp
            + beta_u*cp*jump(theta_w*V(w), n=n)*exnerbar_avg('+')*dS_h_qp
            + beta_u*cp*dot(theta_w*V(w), n)*exnerbar_avg*ds_tb_qp
            - beta_u*cp*div(thetabar_w*w)*exner*dx_qp
            # trace terms appearing after integrating momentum equation
            + beta_u*cp*jump(thetabar_w*w, n=n)*l0('+')*(dS_v_qp + dS_h_qp)
            + beta_u*cp*dot(thetabar_w*w, n)*l0*(ds_tb_qp + ds_v_qp)
            # mass continuity equation
            + (phi*(rho - rho_in) - beta_r*inner(grad(phi), u)*rhobar)*dx
            + beta_r*jump(phi*u, n=n)*rhobar_avg('+')*(dS_v + dS_h)
            # term added because u.n=0 is enforced weakly via the traces
            + beta_r*phi*dot(u, n)*rhobar_avg*(ds_tb + ds_v)
            # constraint equation to enforce continuity of the velocity
            # through the interior facets and weakly impose the no-slip
            # condition
            + dl('+')*jump(u, n=n)*(dS_v + dS_h)
            + dl*dot(u, n)*(ds_t + ds_b + ds_v)
        )

        # TODO: can we get this term using FML?
        # contribution of the sponge term
        if hasattr(self.equations, "mu"):
            eqn += dt*self.equations.mu*inner(w, k)*inner(u, k)*dx_qp

        if equations.parameters.Omega is not None:
            Omega = as_vector([0, 0, equations.parameters.Omega])
            eqn += beta_u*inner(w, cross(2*Omega, u))*dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        appctx = {'slateschur_form': aeqn}

        hybridized_prb = LinearVariationalProblem(aeqn, Leqn, self.y_hybrid,
                                                  constant_jacobian=True)
        self.hybridized_solver = LinearVariationalSolver(hybridized_prb,
                                                    solver_parameters=self.scpc_parameters,
                                                    options_prefix=pc.getOptionsPrefix()+self._prefix,
                                                    appctx=appctx)

        # Project broken u into the HDiv space using facet averaging.
        # Weight function counting the dofs of the HDiv element:
        self._weight = Function(self.Vu)
        weight_kernel = AverageWeightings(self.Vu)
        weight_kernel.apply(self._weight)

        # Averaging kernel
        self._average_kernel = AverageKernel(self.Vu)

        # HDiv-conforming velocity
        self.u_hdiv = Function(self.Vu)

        # Reconstruction of theta
        theta = TrialFunction(self.Vtheta)
        gamma = TestFunction(self.Vtheta)

        self.theta = Function(self.Vtheta)
        theta_eqn = gamma*(theta - theta_in
                           + dot(k, self.u_hdiv)*dot(k, grad(thetabar))*beta_t)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn), rhs(theta_eqn), self.theta,
                                                 constant_jacobian=True)
        cg_ilu_parameters = {'ksp_type': 'cg',
                            'pc_type': 'bjacobi',
                            'sub_pc_type': 'ilu'}
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    solver_parameters=cg_ilu_parameters,
                                                    options_prefix=pc.getOptionsPrefix()+'thetabacksubstitution')
        # Project reference profiles at initialisation
        self.rho_avg_solver.solve()
        self.exner_avg_solver.solve()
        self.hybridized_solver.invalidate_jacobian()



    def apply(self, pc, x, y):
        """
        Apply the preconditioner to x, putting the result in y.

        Args:
            pc (:class:`PETSc.PC`): the preconditioner object.
            x (:class:`PETSc.Vec`): the vector to apply the preconditioner to.
            y (:class:`PETSc.Vec`): the vector to store the result.
        """

        # transfer x -> self.xstar
        with self.xstar.dat.vec_wo as xv:
            x.copy(xv)

        self.xrhs.assign(self.xstar.riesz_representation())
        
        # Solve hybridized system
        self.hybridized_solver.solve()

        # Recover broken u and rho
        u_broken, rho, l = self.y_hybrid.subfunctions
        self.u_hdiv.assign(0)
        self._average_kernel.apply(self.u_hdiv, self._weight, u_broken)
        for bc in self.bcs:
            bc.apply(self.u_hdiv)

        # Transfer data to non-hybrid space
        self.y.subfunctions[0].assign(self.u_hdiv)
        self.y.subfunctions[1].assign(rho)

        # Recover theta
        self.theta.assign(0)
        self.theta_solver.solve()
        self.y.subfunctions[2].assign(self.theta)

        
        with self.y.dat.vec_ro as vout:
            # copy into PETSc output vector
            vout.copy(y)


    def update(self, pc):
        PETSc.Sys.Print("[PC.update] Updating hybridized PC")

        if hasattr(self, "rho_avg_solver"):
            self.rho_avg_solver.solve()
            self.exner_avg_solver.solve()
            self.hybridized_solver.invalidate_jacobian()


    def _process_context(self, pc):
        appctx = self.get_appctx(pc)
        self.appctx = appctx
        self.prefix = pc.getOptionsPrefix() + self._prefix

        self.equations = appctx.get('equations')
        self.mesh = self.equations.domain.mesh

        self.alpha = appctx.get('alpha')
        tau_values = appctx.get('tau_values')
        self.tau_values = tau_values if tau_values is not None else {}

        self.dt = self.equations.domain.dt

    def applyTranspose(self, pc, x, y):
        """
        Apply the transpose of the preconditioner.

        Args:
            pc (:class:`PETSc.PC`): the preconditioner object.
            x (:class:`PETSc.Vec`): the vector to apply the preconditioner to.
            y (:class:`PETSc.Vec`): the vector to put the result into.

        Raises:
            NotImplementedError: this method is currently not implemented.
        """

        raise NotImplementedError("The transpose application of the PC is not implemented.")




