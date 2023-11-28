"""A module containing specialised preconditioners for Gusto applications."""

from firedrake import (dot, jump, dS_h, ds_b, ds_t, ds,
                       FacetNormal, Tensor, AssembledVector)

from firedrake.preconditioners import PCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from gusto.recovery.recovery_kernels import AverageKernel, AverageWeightings
from pyop2.profiling import timed_region, timed_function
from pyop2.utils import as_tuple


__all__ = ["VerticalHybridizationPC"]


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
        from firedrake.assemble import (allocate_matrix, OneFormAssembler,
                                        TwoFormAssembler)
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

        # Magically determine which spaces are vector and scalar valued
        for i, Vi in enumerate(V):

            # Vector-valued spaces will have a non-empty value_shape
            if Vi.ufl_element().value_shape:
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
        Vv_tr = FunctionSpace(mesh, Vv_tr_element)

        # Break the spaces
        broken_elements = MixedElement([BrokenElement(Vi.ufl_element()) for Vi in V])
        V_d = FunctionSpace(mesh, broken_elements)

        # Set up relevant functions
        self.broken_solution = Function(V_d.dual())
        self.broken_residual = Function(V_d)
        self.trace_solution = Function(Vv_tr)
        self.unbroken_solution = Function(V)
        self.unbroken_residual = Function(V)

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
        n = FacetNormal(mesh)
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
                    neumann_subdomains |= set(as_tuple(subdom, int))

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
            markers = [int(x) for x in mesh.exterior_facets.unique_markers]
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
        self._assemble_Srhs = OneFormAssembler(
            K * Atilde.inv * AssembledVector(self.broken_residual),
            tensor=self.schur_rhs,
            form_compiler_parameters=self.ctx.fc_params).assemble

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")

        schur_comp = K * Atilde.inv * K.T
        self.S = allocate_matrix(schur_comp, bcs=trace_bcs,
                                 form_compiler_parameters=self.ctx.fc_params,
                                 mat_type=mat_type,
                                 options_prefix=prefix)
        self._assemble_S = TwoFormAssembler(schur_comp,
                                            tensor=self.S,
                                            bcs=trace_bcs,
                                            form_compiler_parameters=self.ctx.fc_params).assemble

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
        from firedrake.assemble import OneFormAssembler

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
        self._sub_unknown = OneFormAssembler(u_rec,
                                             tensor=u,
                                             form_compiler_parameters=self.ctx.fc_params).assemble

        sigma_rec = A.solve(g - B * AssembledVector(u) - K_0.T * lambdar,
                            decomposition="PartialPivLU")
        self._elim_unknown = OneFormAssembler(sigma_rec,
                                              tensor=sigma,
                                              form_compiler_parameters=self.ctx.fc_params).assemble

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
