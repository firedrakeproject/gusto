from __future__ import absolute_import
import numpy as np
from firedrake import Function, VectorFunctionSpace, SpatialCoordinate, Mesh, \
    FunctionSpace, TensorFunctionSpace, dx, assemble, Constant, split, \
    TrialFunctions, TestFunctions, sqrt, dot, grad, cos, sin, Identity, outer, \
    inner, div, det, TrialFunction, TestFunction, LinearVariationalProblem, \
    LinearVariationalSolver, solve, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, VectorSpaceBasis, MixedVectorSpaceBasis
from gusto.moving_mesh.mesh_generator import MeshGenerator


class MongeAmpereMeshGenerator(MeshGenerator):
    """
    Class for an optimal-transport-based mesh generator

    :arg mesh: mesh for underlying simulation
    :arg monitor: a MonitorFunction object describing the monitor to
    adapt to
    :arg initial_tol: tolerance for initial mesh generation procedure;
    defaults to 1e-4
    :arg tol: tolerance for mesh generation procedure each timestep;
    defaults to 1e-2
    """

    def __init__(self, mesh_in, monitor, initial_tol=1e-4, tol=1e-2):
        self.mesh_in = mesh_in
        self.monitor = monitor
        self.initial_tol = initial_tol
        self.tol = tol

        cellname = mesh_in.ufl_cell().cellname()
        dim = mesh_in.geometric_dimension()

        # Only handle sphere at the moment
        assert dim == 3
        assert cellname in ("triangle", "quadrilateral")
        quads = (cellname == "quadrilateral")

        # Sniff radius of passed-in mesh
        if hasattr(mesh_in, '_icosahedral_sphere_radius'):
            self.R = mesh_in._icoshedral_sphere_radius
        elif hasattr(mesh_in, '_cubed_sphere_radius'):
            self.R = mesh_in._cubed_sphere_radius
        else:
            raise RuntimeError("Mesh doesn't seem to be an IcosahedralSphereMesh or a CubedSphereMesh.  Please make sure you are on the Firedrake branch 'save-cubed-sphere-radius'.")

        # Set up internal 'computational' mesh for own calculations
        new_coords = Function(VectorFunctionSpace(mesh_in, "Q" if quads else "P", 2))
        new_coords.interpolate(SpatialCoordinate(mesh_in))
        new_coords.dat.data[:] *= (self.R / np.linalg.norm(new_coords.dat.data, axis=1)).reshape(-1, 1)
        self.mesh = Mesh(new_coords)

        # Set up copy of passed-in mesh coordinate field for returning to
        # external functions
        self.mesh_coordinates = Function(mesh_in.coordinates)

        # Set up function spaces
        P2 = FunctionSpace(self.mesh, "Q" if quads else "P", 2)
        TensorP2 = TensorFunctionSpace(self.mesh, "Q" if quads else "P", 2)
        MixedSpace = P2*TensorP2
        W_cts = self.mesh.coordinates.function_space()  # guaranteed-continuous version of coordinate field; TODO: change for plane
        P1 = FunctionSpace(self.mesh, "Q" if quads else "P", 1)  # for representing m

        if dim == 3:  # sphere
            dxdeg = dx(degree=8)  # quadrature degree for nasty terms
        else:
            dxdeg = dx

        # get mesh area
        self.total_area = assemble(Constant(1.0)*dxdeg(domain=mesh))

        # Set up functions
        self.phisigma = Function(MixedSpace)
        self.phi, self.sigma = split(self.phisigma)
        self.phisigma_temp = Function(MixedSpace)
        self.phi_temp, self.sigma_temp = split(self.phisigma_temp)

        # mesh coordinates
        self.x = Function(mesh.coordinates)  # for 'physical' coords
        self.x_old = Function(mesh.coordinates)
        self.xi = Function(mesh.coordinates)  # for 'computational' coords

        self.theta = Constant(0.0)

        # Define mesh equations
        v, tau = TestFunctions(MixedSpace)

        # sphere
        modgphi = sqrt(dot(grad(self.phi), grad(self.phi)) + 1e-8)
        expxi = xi*cos(modgphi) + grad(self.phi)*sin(modgphi)/modgphi
        projxi = Identity(3) - outer(self.xi, self.xi)

        modgphi_temp = sqrt(dot(grad(self.phi_temp), grad(self.phi_temp)) + 1e-8)
        expxi_temp = self.xi*cos(modgphi_temp) + grad(self.phi_temp)*sin(modgphi_temp)/modgphi_temp

        F_mesh = inner(self.sigma, tau)*dxdeg + dot(div(tau), expxi)*dxdeg - (self.m*det(outer(expxi, self.xi) + dot(self.sigma, projxi)) - self.theta)*v*dxdeg
        self.thetaform = self.m*det(outer(expxi_temp, self.xi) + dot(self.sigma_temp, projxi))*dxdeg

        # Define a solver for obtaining grad(phi) by L^2 projection
        # TODO: lump this?
        u_cts = TrialFunction(W_cts)
        v_cts = TestFunction(W_cts)

        self.gradphi_cts = Function(W_cts)

        a_cts = dot(v_cts, u_cts)*dx
        L_gradphi = dot(v_cts, grad(self.phi_temp))*dx

        probgradphi = LinearVariationalProblem(a_cts, L_gradphi, self.gradphi_cts)
        self.solvgradphi = LinearVariationalSolver(probgradphi, solver_parameters={'ksp_type': 'cg'})

        if False:
            # plane
            gradphi_dg = Function(mesh.coordinates).assign(0)
        if True:
            # sphere
            gradphi_cts2 = Function(W_cts)  # extra, as gradphi_cts not necessarily tangential

        # Set up initial sigma on sphere
        sigma_ = TrialFunction(TensorP2)
        tau_ = TestFunction(TensorP2)
        sigma_ini = Function(TensorP2)

        asigmainit = inner(sigma_, tau_)*dxdeg
        if False:
            # plane
            Lsigmainit = -dot(div(tau_), grad(phi))*dx
        else:
            # sphere
            Lsigmainit = -dot(div(tau_), expxi)*dxdeg

        solve(asigmainit == Lsigmainit, sigma_ini, solver_parameters={'ksp_type': 'cg'})

        self.phisigma.sub(1).assign(sigma_ini)

        # Solver options for mesh generation
        phi__, sigma__ = TrialFunctions(MixedSpace)
        v__, tau__ = TestFunctions(MixedSpace)

        # Custom preconditioning matrix
        Jp = inner(sigma__, tau__)*dx + phi__*v__*dx + dot(grad(phi__), grad(v__))*dx

        mesh_prob = NonlinearVariationalProblem(F_mesh, phisigma, Jp=Jp)
        V1_nullspace = VectorSpaceBasis(constant=True)
        nullspace = MixedVectorSpaceBasis(MixedSpace, [V1_nullspace, MixedSpace.sub(1)])

        params = {"ksp_type": "gmres",
                  "pc_type": "fieldsplit",
                  "pc_fieldsplit_type": "multiplicative",
                  "pc_fieldsplit_off_diag_use_amat": True,
                  "fieldsplit_0_pc_type": "gamg",
                  "fieldsplit_0_ksp_type": "preonly",
                  "fieldsplit_0_mg_levels_ksp_max_it": 5,
                  "fieldsplit_0_mg_levels_pc_type": "ilu",
                  "fieldsplit_1_pc_type": "ilu",
                  "fieldsplit_1_ksp_type": "preonly",
                  "ksp_max_it": 200,
                  "snes_max_it": 50,
                  "ksp_gmres_restart": 200,
                  "snes_rtol": initial_tol,
                  "snes_linesearch_type": "bt",
                  # "ksp_monitor": True,
                  # "snes_monitor": True,
                  # "snes_linesearch_monitor": True,
                  "snes_lag_preconditioner": -1}

        self.mesh_solv = NonlinearVariationalSolver(mesh_prob,
                                                    nullspace=nullspace,
                                                    transpose_nullspace=nullspace,
                                                    pre_jacobian_callback=self.update_mxtheta,
                                                    pre_function_callback=self.update_mxtheta,
                                                    solver_parameters=params)

        self.mesh_solv.snes.setMonitor(fakemonitor)

    def firstrun(self, foo):
        """
        This function adapts the mesh to the initial state.

        :arg ???: ???
        :arg ???: ???
        """
        assert not self.initial_mesh
        self.mesh_solv.solve()
        self.initial_mesh = False
        self.mesh.coordinates.assign(x)  # ???

    def generate_mesh(self, foo):
        # Back up the current mesh
        self.x_old.assign(x)

        # Obtain monitor function on old mesh
        self.get_m_from_q(q)
        self.m_old.assign(m)

        # Generate new mesh, coords go into x
        mesh_solv.solve()

    def update_mxtheta(self, cursol):
        with self.phisigma_temp.dat.vec as v:
            cursol.copy(v)

        # Obtain continous version of grad phi.
        self.mesh.coordinates.assign(xi)
        self.solvgradphi.solve()

        # Generate coordinates from this.
        if False:
            # On plane, simply copy into the discontinous coordinate field
            par_loop("""
for (int i=0; i<cg.dofs; i++) {
    for (int j=0; j<2; j++) {
        dg[i][j] = cg[i][j];
    }
}
""", dx, {'cg': (gradphi_cts, READ),
          'dg': (gradphi_dg, WRITE)})

            x.assign(xi + gradphi_dg)  # x = xi + grad(phi)
        else:
            # On sphere, firstly "fix grad(phi)" by ensuring that
            # grad(phi).x = 0, assuming |x| = 1
            par_loop("""
for (int i=0; i<vnew.dofs; i++) {
    double dot = 0.0;
    for (int j=0; j<3; j++) {
        dot += x[i][j]*v[i][j];
    }
    for (int j=0; j<3; j++) {
        vnew[i][j] = v[i][j] - dot*x[i][j];
    }
}
""", dx, {'x': (mesh.coordinates, READ),
          'v': (gradphi_cts, READ),
          'vnew': (gradphi_cts2, WRITE)})

            # Then use exponential map to obtain x
            par_loop("""
for (int i=0; i<xi.dofs; i++) {
    double norm = 0.0;
    for (int j=0; j<3; j++) {
        norm += u[i][j]*u[i][j];
    }
    norm = sqrt(norm) + 1e-8;

    for (int j=0; j<3; j++) {
        x[i][j] = xi[i][j]*cos(norm) + (u[i][j]/norm)*sin(norm);
    }
}
""", dx, {'xi': (xi, READ),
          'u': (gradphi_cts2, READ),
          'x': (x, WRITE)})

        if self.initial_mesh:
            # q obtained by interpolation on successive meshes,
            # and m is obtained directly from that

            self.mesh.coordinates.assign(x)
            assert False
            q.interpolate(q_expr)
            get_m_from_q(q)
            if uniform:
                m.assign(1.0)
        else:
            # We have the function m on old mesh: m_old. We want to represent
            # this on the trial mesh. Do this by using the same values
            # (equivalent to advection by +v), then do an advection step of -v.

            mesh.coordinates.assign(x)
            mesh_adv_vel.assign(x_old - x)

            # Make discontinuous m
            par_loop("""
for (int i=0; i<dg.dofs; i++) {
    dg[i][0] = cg[i][0];
}
""", dx, {'dg': (m_dg, WRITE),
          'cg': (m_old, READ)})

            # Advect this by -v
            steps = 5
            for ii in range(steps):
                solv_madv.solve()
                m_dg.assign(m_dg + Constant(1.0/steps)*dm)
                project(m_dg, mbar)  # get centroids for slope limiter
                limit_slope(m_dg, mbar, m_max, m_min)

            project(m_dg, m)  # project discontinuous m back into CG
            mmax_post = max(m.dat.data)/min(m.dat.data)

        mesh.coordinates.assign(xi)
        theta_new = assemble(thetaform)/total_area
        theta.assign(theta_new)


    def fakemonitor(self, snes, it, rnorm):
        cursol = snes.getSolution()
        self.update_mxtheta(cursol)  # updates m, x, and theta
