from __future__ import absolute_import
import numpy as np
from mpi4py import MPI
from firedrake import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, \
    Function, Constant, dx, dS, assemble, TrialFunction, TestFunction, sqrt, \
    dot, inner, FacetNormal, jump, grad, div, as_vector, \
    LinearVariationalProblem, LinearVariationalSolver, LinearSolver, par_loop, \
    RW, READ, Mesh


class MonitorFunction(object):
    """
    Base class for monitor function generation

    :arg f: the Function that will be used to generate a monitor function
    :arg adapt_to: property of the field to adapt to; must be
    "function" [default], "gradient" or "hessian"
    :arg avg_weight: parameter for monitor function regularisation. The raw
    monitor function m is replaced by avg_weight*m-avg + (1 - avg_weight)*m.
    Set to 0.0 to disable. Defaults to 0.5.
    :arg max_min_cap: parameter that causes m to be clipped from above so that
    the ratio between the smallest and largest monitor function values is no
    more than this number. Set to 0.0 to disable. Defaults to 4.0.
    """

    def __init__(self, f, adapt_to="function", avg_weight=0.5, max_min_cap=4.0):

        assert adapt_to in ("function", "gradient", "hessian")
        assert 0.0 <= avg_weight < 1.0
        assert max_min_cap >= 1.0 or max_min_cap == 0.0

        assert f.ufl_shape == ()  # can lift this restriction later

        self.adapt_to = adapt_to
        self.avg_weight = avg_weight
        self.max_min_cap = max_min_cap

        cellname = f.ufl_element().cell().cellname()
        dim = f.ufl_element().cell().geometric_dimension()

        assert cellname in ("triangle", "quadrilateral")
        assert dim in (2, 3)

        quads = (cellname == "quadrilateral")

        # Set up internal mesh for monitor function calculations
        new_coords = Function(f.function_space().mesh().coordinates)
        self.mesh = Mesh(new_coords)

        self.user_f = f  # store 'pointer' to original function
        self.f = Function(FunctionSpace(self.mesh, f.ufl_element()))  # make "own copy" of f on internal mesh

        P1 = FunctionSpace(self.mesh, "Q" if quads else "P", 1)  # for representing m
        P0 = FunctionSpace(self.mesh, "DQ" if quads else "DP", 0)  # slope limiter centroids
        DP1 = FunctionSpace(self.mesh, "DQ" if quads else "DP", 1)  # for advection

        if self.adapt_to in ("gradient", "hessian"):
            VectorP1 = VectorFunctionSpace(self.mesh, "Q" if quads else "P", 1)
            self.gradf = Function(VectorP1)
            self.gradf_lhs = Function(VectorP1)
            if self.adapt_to == "hessian":
                TensorP1 = TensorFunctionSpace(self.mesh, "Q" if quads else "P", 1)
                self.hessf = Function(TensorP1)
                self.hessf_lhs = Function(TensorP1)

        self.mesh_adv_vel = Function(self.mesh.coordinates)

        # monitor function and related quantities
        self.m = Function(P1)
        self.m_prereg = Function(P1)
        self.m_old = Function(P1)

        self.m_dg = Function(DP1)
        self.dm = Function(DP1)

        self.mbar = Function(P0)
        self.m_max = Function(P1)
        self.m_min = Function(P1)

        if avg_weight > 0.0:
            self.m_integral = self.m_prereg*dx
            self.total_area = assemble(Constant(1.0)*dx(self.mesh))

        # Set up monitor generation equations
        # TODO: generalise to vector-valued f

        if self.adapt_to in ("gradient", "hessian"):
            # Forms for lumped weak gradient of f
            v_vp1 = TestFunction(VectorP1)
            v_ones = as_vector(np.ones(dim))
            self.a_vp1_lumped = dot(v_vp1, v_ones)*dx
            self.L_vp1 = -div(v_vp1)*self.f*dx

            if self.adapt_to == "hessian":
                # Forms for lumped hessian of f
                t_ones = as_vector(np.ones((dim, dim)))
                v_tp1 = TestFunction(TensorP1)
                self.a_tp1_lumped = inner(v_tp1, t_ones)*dx
                self.L_tp1 = inner(v_tp1, grad(self.gradf))*dx

        # Forms for lumped projection of monitor function into P1
        v_p1 = TestFunction(P1)
        self.a_p1_lumped = v_p1*dx
        self.L_monitor_rhs = Function(P1)
        self.L_monitor_lhs = Function(P1)
        if adapt_to == "function":
            self.L_monitor = v_p1*sqrt(inner(self.f, self.f))*dx
        elif adapt_to == "gradient":
            self.L_monitor = v_p1*sqrt(inner(self.gradf, self.gradf))*dx
        elif adapt_to == "hessian":
            self.L_monitor = v_p1*sqrt(inner(self.hessf, self.hessf))*dx

        # Define mesh 'backwards advection'
        u_dg = TrialFunction(DP1)
        v_dg = TestFunction(DP1)
        n = FacetNormal(self.mesh)

        self.a_dg = v_dg*u_dg*dx
        vn = 0.5*(dot(self.mesh_adv_vel, n) + abs(dot(self.mesh_adv_vel, n)))
        self.L_madv = self.m_dg*div(v_dg*self.mesh_adv_vel)*dx - jump(v_dg)*jump(vn*self.m_dg)*dS
        self.L_madv_fn = Function(DP1)
        self.a_dg_mat = assemble(self.a_dg)

        # Set up forms for centroid-generation (slope limiter)
        u_p0 = TrialFunction(P0)
        v_p0 = TestFunction(P0)
        self.a_p0 = v_p0*u_p0*dx
        self.L_p0 = v_p0*self.m_dg*dx
        self.a_p0_mat = assemble(self.a_p0)
        self.a_p0_rhs = Function(P0)

        # Set up forms for projection of DG m back into P1
        u_p1 = TrialFunction(P1)
        v_p1 = TestFunction(P1)
        a_p1 = v_p1*u_p1*dx
        L_p1 = v_p1*self.m_dg*dx

        prob_m_p1_proj = LinearVariationalProblem(a_p1, L_p1, self.m, constant_jacobian=False)
        self.solv_m_p1_proj = LinearVariationalSolver(prob_m_p1_proj, solver_parameters={'ksp_type': 'cg'})

    def update_monitor(self):
        self.f.dat.data[:] = self.user_f.dat.data[:]

        if self.adapt_to in ("gradient", "hessian"):
            # Form mass-lumped gradient
            assemble(self.L_vp1, tensor=self.gradf)
            assemble(self.a_vp1_lumped, tensor=self.gradf_lhs)
            self.gradf.dat /= self.gradf_lhs.dat

            if self.adapt_to == "hessian":
                # Form mass-lumped hessian
                assemble(self.L_tp1, tensor=self.hessf)
                assemble(self.a_tp1_lumped, tensor=self.hessf_lhs)
                self.hessf.dat /= self.hessf_lhs.dat

        # obtain m (pre-regularisation) by lumped projection
        assemble(self.L_monitor, tensor=self.L_monitor_rhs)
        assemble(self.a_p1_lumped, tensor=self.L_monitor_lhs)
        self.m_prereg.dat.data[:] = self.L_monitor_rhs.dat.data_ro[:] / self.L_monitor_lhs.dat.data_ro[:]

        # regularise

        # if both disabled, do nothing
        if self.avg_weight == 0.0 and self.max_min_cap == 0.0:
            self.m.assign(self.m_prereg)

        # if no Beckett/Mackenzie, just cap
        elif self.avg_weight == 0.0:
            m_prereg_min = self.m_prereg.comm.allreduce(self.m_prereg.dat.data_ro.min(), op=MPI.MIN)
            self.m.dat.data[:] = np.fmin(self.m_prereg.dat.data_ro[:], self.max_min_cap*m_prereg_min)

        # if no max/min cap, just Beckett/Mackenzie
        elif self.max_min_cap == 0.0:
            m_int = assemble(self.m_integral)
            m_avg = m_int/self.total_area
            self.m.assign(Constant(self.avg_weight)*Constant(m_avg) + Constant(1.0 - self.avg_weight)*self.m_prereg)

        # otherwise, do both
        else:
            # consider doing this step twice (or more), since m_avg will
            # decrease after m_prereg is capped
            for ii in range(4):
                m_int = assemble(self.m_integral)
                m_avg = m_int/self.total_area
                self.m_prereg.dat.data[:] = np.fmin(self.m_prereg.dat.data[:], (self.max_min_cap - 1.0)*(self.avg_weight/(1.0 - self.avg_weight))*m_avg)
                print m_int, assemble(self.m_integral)
            print

            self.m.assign(Constant(self.avg_weight)*Constant(m_avg) + Constant(1.0 - self.avg_weight)*self.m_prereg)

        # safety check for now
        assert (self.m.dat.data_ro >= 0.0).all()

        # make m O(1)
        m_min = self.m.comm.allreduce(self.m.dat.data_ro.min(), op=MPI.MIN)
        self.m.dat.data[:] /= m_min

        # debugging
        # print "after creation:", np.min(self.m.dat.data_ro), np.max(self.m.dat.data_ro)

        # return the monitor function, or do nothing?
        # return self.m

    def get_monitor_on_new_mesh(self, m, x_old, x_new):
        # We have the function m on old mesh: self.m_old. We want to represent
        # this on the new mesh. Do this by using the same values
        # (equivalent to advection by +v), then do an advection step of -v.

        self.mesh.coordinates.assign(x_new)
        self.mesh_adv_vel.assign(x_old - x_new)

        # Make discontinuous m
        self.m_dg.interpolate(self.m_old)

        # Set up solver. Do this here rather than setting
        # constant_jacobian=False, since doing several steps
        assemble(self.a_dg, tensor=self.a_dg_mat)
        A = LinearSolver(self.a_dg_mat,
                         solver_parameters={'ksp_type': 'preonly',
                                            'pc_type': 'bjacobi',
                                            'sub_pc_type': 'ilu'})

        assemble(self.a_p0, tensor=self.a_p0_mat)
        A_p0 = LinearSolver(self.a_p0_mat,
                            solver_parameters={'ksp_type': 'preonly',
                                               'pc_type': 'bjacobi',
                                               'sub_pc_type': 'ilu'})

        # Advect m by -v
        steps = 5
        for ii in range(steps):
            # assemble and solve advection step
            assemble(self.L_madv, tensor=self.L_madv_fn)
            A.solve(self.dm, self.L_madv_fn)

            self.m_dg.assign(self.m_dg + Constant(1.0/steps)*self.dm)

            # obtain centroids for slope limiting
            assemble(self.L_p0, tensor=self.a_p0_rhs)
            A_p0.solve(self.mbar, self.a_p0_rhs)

            # perform slope limiting
            self._limit_slope(self.m_dg, self.mbar, self.m_max, self.m_min)

        # project self.m_dg back into P1 as self.m
        self.solv_m_p1_proj.solve()

        # print "after advection:", np.min(self.m.dat.data_ro), np.max(self.m.dat.data_ro)

        # safety check for now
        assert (self.m.dat.data_ro >= 0.0).all()

    def _limit_slope(self, field, centroids, max_field, min_field):
        max_field.assign(-1e300)  # small number
        min_field.assign(1e300)  # big number

        # Set up fields containing max/min of neighbouring cell averages
        par_loop("""
for (int i=0; i<qmax.dofs; i++) {
    qmax[i][0] = fmax(qmax[i][0], centroids[0][0]);
    qmin[i][0] = fmin(qmin[i][0], centroids[0][0]);
}
""", dx, {'qmax': (max_field, RW),
          'qmin': (min_field, RW),
          'centroids': (centroids, READ)})

        # Limit DG field values to be between max and min
        par_loop("""
double alpha = 1.0;
double cellavg = centroids[0][0];
for (int i=0; i<q.dofs; i++) {
    if (q[i][0] > cellavg)
        alpha = fmin(alpha, fmin(1, (qmax[i][0] - cellavg)/(q[i][0] - cellavg)));
    else if (q[i][0] < cellavg)
        alpha = fmin(alpha, fmin(1, (cellavg - qmin[i][0])/(cellavg - q[i][0])));
}
for (int i=0; i<q.dofs; i++) {
    q[i][0] = cellavg + alpha*(q[i][0] - cellavg);
}
""", dx, {'q': (field, RW),
          'qmax': (max_field, READ),
          'qmin': (min_field, READ),
          'centroids': (centroids, READ)})
