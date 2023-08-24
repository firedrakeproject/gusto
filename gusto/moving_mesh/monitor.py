from firedrake import *
import numpy as np
from mpi4py import MPI

__all__ = ["MonitorFunction"]


class MonitorFunction(object):

    def __init__(self, f_name, adapt_to="function", avg_weight=0.5,
                 max_min_cap=4.0):

        assert adapt_to in ("function", "gradient", "hessian")
        assert 0.0 <= avg_weight < 1.0
        assert max_min_cap >= 1.0 or max_min_cap == 0.0

        self.f_name = f_name
        self.adapt_to = adapt_to
        self.avg_weight = avg_weight
        self.max_min_cap = max_min_cap

    def setup(self, state_fields):
        f = state_fields(self.f_name)
        cellname = f.ufl_element().cell().cellname()
        quads = (cellname == "quadrilateral")

        # Set up internal mesh for monitor function calculations
        new_coords = Function(f.function_space().mesh().coordinates)
        self.mesh = Mesh(new_coords)
        self.f = Function(FunctionSpace(self.mesh, f.ufl_element()))  # make "own copy" of f on internal mesh
        self.user_f = f

        # Set up function spaces
        P1 = FunctionSpace(self.mesh, "Q" if quads else "P", 1)  # for representing m
        DP1 = FunctionSpace(self.mesh, "DQ" if quads else "DP", 1)  # for advection
        VectorP1 = VectorFunctionSpace(self.mesh, "Q" if quads else "P", 1)
        self.limiter = VertexBasedLimiter(DP1)

        self.gradq = Function(VectorP1)
        if self.adapt_to == "hessian":
            TensorP1 = TensorFunctionSpace(self.mesh, "Q" if quads else "P", 1)
            self.hessq = Function(TensorP1)

        # get mesh area
        self.total_area = assemble(Constant(1.0)*dx(self.mesh))

        self.m = Function(P1)
        self.m_prereg = Function(P1)
        self.m_old = Function(P1)
        self.m_dg = Function(DP1)
        self.dm = Function(DP1)
        self.m_int_form = self.m_prereg*dx

        # define monitor function in terms of q
        if False:   # for plane
            v_ones = as_vector(np.ones(2))
        else:
            v_ones = as_vector(np.ones(3))
        # Obtain weak gradient
        # u_vp1 = TrialFunction(VectorP1)
        v_vp1 = TestFunction(VectorP1)
        self.a_vp1_lumped = dot(v_vp1, v_ones)*dx
        self.L_vp1 = -div(v_vp1)*self.f*dx

        if self.adapt_to == "hessian":
            if False:   # for plane
                t_ones = as_vector(np.ones((2, 2)))
            else:
                t_ones = as_vector(np.ones((3, 3)))
            # Obtain approximation to hessian
            # u_tp1 = TrialFunction(TensorP1)
            v_tp1 = TestFunction(TensorP1)
            self.a_tp1_lumped = inner(v_tp1, t_ones)*dx
            self.L_tp1 = inner(v_tp1, grad(self.gradq))*dx

        # Define forms for lumped project of monitor function into P1
        v_p1 = TestFunction(P1)
        self.a_p1_lumped = v_p1*dx
        if True:   # not uniform:
            if self.adapt_to == "gradient":
                self.L_monitor = v_p1*sqrt(dot(self.gradq, self.gradq))*dx
            elif self.adapt_to == "hessian":
                self.L_monitor = v_p1*sqrt(inner(self.hessq, self.hessq))*dx
            else:
                self.L_monitor = v_p1*dx

        u_dg = TrialFunction(DP1)
        v_dg = TestFunction(DP1)
        n = FacetNormal(self.mesh)
        self.mesh_adv_vel = Function(self.mesh.coordinates)
        vn = 0.5*(dot(self.mesh_adv_vel, n) + abs(dot(self.mesh_adv_vel, n)))
        a_dg = v_dg*u_dg*dx
        L_madv = self.m_dg*div(v_dg*self.mesh_adv_vel)*dx - jump(v_dg)*jump(vn*self.m_dg)*dS

        prob_madv = LinearVariationalProblem(a_dg, L_madv, self.dm, constant_jacobian=False)
        self.solv_madv = LinearVariationalSolver(prob_madv,
                                                 solver_parameters={"ksp_type": "preonly",
                                                                    "pc_type": "bjacobi",
                                                                    "sub_ksp_type": "preonly",
                                                                    "sub_pc_type": "ilu"})

    def update_monitor(self):

        self.f.dat.data[:] = self.user_f.dat.data[:]

        # TODO don't need to do this if adapting to function
        assemble(self.L_vp1, tensor=self.gradq)
        self.gradq.dat /= assemble(self.a_vp1_lumped).dat  # obtain weak gradient
        if self.adapt_to == "hessian":
            assemble(self.L_tp1, tensor=self.hessq)
            self.hessq.dat /= assemble(self.a_tp1_lumped).dat

        # obtain m by lumped projection
        self.m_prereg.interpolate(assemble(self.L_monitor)/assemble(self.a_p1_lumped))

        # calculate average of m
        m_int = assemble(self.m_int_form)
        m_avg = m_int/self.total_area

        # cap max-to-avg ratio
        self.m_prereg.dat.data[:] = np.fmin(self.m_prereg.dat.data[:], (self.max_min_cap - 1.0)*m_avg)

        # use Beckett+Mackenzie regularization
        self.m.assign(Constant(self.avg_weight)*self.m_prereg + Constant(1.0 - self.avg_weight)*Constant(m_avg))

        assert (self.m.dat.data >= 0.0).all()

        # make m O(1)
        m_min = self.m.comm.allreduce(self.m.dat.data_ro.min(), op=MPI.MIN)
        self.m.dat.data[:] /= m_min

        # mmax_pre = max(self.m.dat.data)/min(self.m.dat.data)

    def get_monitor_on_new_mesh(self, m, x_old, x_new):
        # We have the function m on old mesh: m_old. We want to represent
        # this on the trial mesh. Do this by using the same values
        # (equivalent to advection by +v), then do an advection step of -v.

        self.mesh.coordinates.assign(x_new)
        self.limiter.centroid_solver = self.limiter._construct_centroid_solver()
        self.mesh_adv_vel.assign(x_old - x_new)

        # Make discontinuous m
        self.m_dg.interpolate(self.m_old)

        # Advect this by -v
        for ii in range(10):
            self.solv_madv.solve()
            self.m_dg.assign(self.m_dg + Constant(1.0/10.)*self.dm)
            self.limiter.apply(self.m_dg)

        project(self.m_dg, self.m)  # project discontinuous m back into CG
        # mmax_post = max(self.m.dat.data)/min(self.m.dat.data)
