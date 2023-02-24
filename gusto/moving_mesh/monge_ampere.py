from firedrake import *
import numpy as np

class OptimalTransportMeshGenerator(object):

    def __init__(self, mesh_in, monitor, initial_tol=1.e-4, tol=1.e-2, pre_meshgen_callback=None, post_meshgen_callback=None):

        self.mesh_in = mesh_in
        self.monitor = monitor
        self.initial_tol = initial_tol
        self.tol = tol
        self.pre_meshgen_fn = pre_meshgen_callback
        self.post_meshgen_fn = post_meshgen_callback

    def setup(self):
        mesh_in = self.mesh_in
        cellname = mesh_in.ufl_cell().cellname()
        quads = (cellname == "quadrilateral")

        if hasattr(mesh_in, '_radius'):
            self.meshtype = "sphere"
            self.R = mesh_in._radius
            self.Rc = Constant(self.R)
            dxdeg = dx(degree=8)  # quadrature degree for nasty terms
        else:
            self.meshtype = "plane"
            dxdeg = dx

        # Set up internal 'computational' mesh for own calculations
        # This will be a unit sphere; make sure to scale appropriately
        # when bringing coords in and out.
        new_coords = Function(VectorFunctionSpace(mesh_in, "Q" if quads else "P", 2))
        new_coords.interpolate(SpatialCoordinate(mesh_in))
        new_coords.dat.data[:] /= np.linalg.norm(new_coords.dat.data, axis=1).reshape(-1, 1)
        self.mesh = Mesh(new_coords)

        # Set up copy of passed-in mesh coordinate field for returning to
        # external functions
        self.output_coords = Function(mesh_in.coordinates)
        self.own_output_coords = Function(VectorFunctionSpace(self.mesh, "Q" if quads else "P", mesh_in.coordinates.ufl_element().degree()))

        # get mesh area
        self.total_area = assemble(Constant(1.0)*dx(self.mesh))

        ### SET UP FUNCTIONS ###
        P2 = FunctionSpace(self.mesh, "Q" if quads else "P", 2)
        TensorP2 = TensorFunctionSpace(self.mesh, "Q" if quads else "P", 2)
        MixedSpace = P2*TensorP2
        P1 = FunctionSpace(self.mesh, "Q" if quads else "P", 1)  # for representing m
        W_cts = VectorFunctionSpace(self.mesh, "Q" if quads else "P", 1 if self.meshtype == "plane" else 2)  # guaranteed-continuous version of coordinate field

        self.phisigma = Function(MixedSpace)
        self.phi, self.sigma = split(self.phisigma)
        self.phisigma_temp = Function(MixedSpace)
        self.phi_temp, self.sigma_temp = split(self.phisigma_temp)

        # mesh coordinates
        self.x = Function(self.mesh.coordinates)  # for 'physical' coords
        self.xi = Function(self.mesh.coordinates)  # for 'computational' coords

        # monitor function mesh coords
        self.x_old = Function(self.monitor.mesh.coordinates)
        self.x_new = Function(self.monitor.mesh.coordinates)

        self.m = Function(P1)
        self.theta = Constant(0.0)

        ### DEFINE MESH EQUATIONS ###
        v, tau = TestFunctions(MixedSpace)

        if self.meshtype == "plane":
            I = Identity(2)
            F_mesh = inner(self.sigma, tau)*dx + dot(div(tau), grad(self.phi))*dx - (self.m*det(I + self.sigma) - self.theta)*v*dx
            self.thetaform = self.m*det(I + self.sigma_temp)*dx

        elif self.meshtype == "sphere":
            modgphi = sqrt(dot(grad(self.phi), grad(self.phi)) + 1e-12)
            expxi = self.xi*cos(modgphi) + grad(self.phi)*sin(modgphi)/modgphi
            projxi = Identity(3) - outer(self.xi, self.xi)

            modgphi_temp = sqrt(dot(grad(self.phi_temp), grad(self.phi_temp)) + 1e-12)
            expxi_temp = self.xi*cos(modgphi_temp) + grad(self.phi_temp)*sin(modgphi_temp)/modgphi_temp

            F_mesh = inner(self.sigma, tau)*dxdeg + dot(div(tau), expxi)*dxdeg - (self.m*det(outer(expxi, self.xi) + dot(self.sigma, projxi)) - self.theta)*v*dxdeg
            self.thetaform = self.m*det(outer(expxi_temp, self.xi) + dot(self.sigma_temp, projxi))*dxdeg

        # Define a solver for obtaining grad(phi) by L^2 projection
        u_cts = TrialFunction(W_cts)
        v_cts = TestFunction(W_cts)

        self.gradphi_cts = Function(W_cts)

        a_cts = dot(v_cts, u_cts)*dx
        L_gradphi = dot(v_cts, grad(self.phi_temp))*dx

        probgradphi = LinearVariationalProblem(a_cts, L_gradphi, self.gradphi_cts)
        self.solvgradphi = LinearVariationalSolver(probgradphi, solver_parameters={'ksp_type': 'cg'})

        if self.meshtype == "plane":
            self.gradphi_dg = Function(mesh.coordinates).assign(0)
        elif self.meshtype == "sphere":
            self.gradphi_cts2 = Function(W_cts)  # extra, as gradphi_cts not necessarily tangential

        ### SET UP INITIAL SIGMA ### (needed on sphere)
        sigma_ = TrialFunction(TensorP2)
        tau_ = TestFunction(TensorP2)
        sigma_ini = Function(TensorP2)

        asigmainit = inner(sigma_, tau_)*dxdeg
        if self.meshtype == "plane":
            Lsigmainit = -dot(div(tau_), grad(self.phi))*dx
        else:
            Lsigmainit = -dot(div(tau_), expxi)*dxdeg

        solve(asigmainit == Lsigmainit, sigma_ini, solver_parameters={'ksp_type': 'cg'})

        self.phisigma.sub(1).assign(sigma_ini)

        ### SOLVER OPTIONS FOR MESH GENERATION ###
        phi__, sigma__ = TrialFunctions(MixedSpace)
        v__, tau__ = TestFunctions(MixedSpace)

        # Custom preconditioning matrix
        Jp = inner(sigma__, tau__)*dx + phi__*v__*dx + dot(grad(phi__), grad(v__))*dx

        self.mesh_prob = NonlinearVariationalProblem(F_mesh, self.phisigma, Jp=Jp)
        V1_nullspace = VectorSpaceBasis(constant=True)
        self.nullspace = MixedVectorSpaceBasis(MixedSpace, [V1_nullspace, MixedSpace.sub(1)])

        self.params = {"ksp_type": "gmres",
                       "pc_type": "fieldsplit",
                       "pc_fieldsplit_type": "multiplicative",
                       "pc_fieldsplit_off_diag_use_amat": True,
                       "fieldsplit_0_pc_type": "gamg",
                       "fieldsplit_0_ksp_type": "preonly",
                       # "fieldsplit_0_mg_levels_ksp_type": "chebyshev",
                       # "fieldsplit_0_mg_levels_ksp_chebyshev_estimate_eigenvalues": True,
                       # "fieldsplit_0_mg_levels_ksp_chebyshev_estimate_eigenvalues_random": True,
                       "fieldsplit_0_mg_levels_ksp_max_it": 5,
                       "fieldsplit_0_mg_levels_pc_type": "bjacobi",
                       "fieldsplit_0_mg_levels_sub_ksp_type": "preonly",
                       "fieldsplit_0_mg_levels_sub_pc_type": "ilu",
                       "fieldsplit_1_ksp_type": "preonly",
                       "fieldsplit_1_pc_type": "bjacobi",
                       "fieldsplit_1_sub_ksp_type": "preonly",
                       "fieldsplit_1_sub_pc_type": "ilu",
                       "ksp_max_it": 100,
                       "snes_max_it": 50,
                       "ksp_gmres_restart": 100,
                       "snes_rtol": self.initial_tol,
                       # "snes_atol": 1e-12,
                       "snes_linesearch_type": "l2",
                       "snes_linesearch_max_it": 5,
                       "snes_linesearch_maxstep": 1.05,
                       "snes_linesearch_damping": 0.8,
                       # "ksp_monitor": None,
                       "snes_monitor": None,
                       # "snes_linesearch_monitor": None,
                       "snes_lag_preconditioner": -2}

        self.mesh_solv = NonlinearVariationalSolver(
            self.mesh_prob,
            nullspace=self.nullspace,
            transpose_nullspace=self.nullspace,
            pre_jacobian_callback=self.update_mxtheta,
            pre_function_callback=self.update_mxtheta,
            solver_parameters=self.params)

        self.mesh_solv.snes.setMonitor(self.fakemonitor)

    def update_mxtheta(self, cursol):
        with self.phisigma_temp.dat.vec as v:
            cursol.copy(v)

        # Obtain continous version of grad phi.
        self.mesh.coordinates.assign(self.xi)
        self.solvgradphi.solve()

        # "Fix grad(phi) on sphere"
        if self.meshtype == "sphere":
            # Ensures that grad(phi).x = 0, assuming |x| = 1
            # v_new = v - (v.x)x
            self.gradphi_cts2.interpolate(self.gradphi_cts - dot(self.gradphi_cts, self.mesh.coordinates)*self.mesh.coordinates)

        # Generate coordinates
        if self.meshtype == "plane":
            self.gradphi_dg.interpolate(self.gradphi_cts)
            self.x.assign(self.xi + self.gradphi_dg)  # x = xi + grad(phi)
        else:
            # Generate new coordinate field using exponential map
            # x = cos(|v|)*x + sin(|v|)*(v/|v|)
            gradphinorm = sqrt(dot(self.gradphi_cts2, self.gradphi_cts2)) + 1e-12
            self.x.interpolate(cos(gradphinorm)*self.xi + sin(gradphinorm)*(self.gradphi_cts2/gradphinorm))

        if self.initial_mesh:
            # self.mesh.coordinates.assign(self.x)
            self.own_output_coords.interpolate(Constant(self.R)*self.x)
            self.output_coords.dat.data[:] = self.own_output_coords.dat.data_ro[:]
            self.mesh_in.coordinates.assign(self.output_coords)
            #self.mesh_in.coordinates.assign(self.x)
            self.initialise_fn()
            #self.monitor.mesh.coordinates.dat.data[:] = self.mesh.coordinates.dat.data_ro[:]
            self.monitor.mesh.coordinates.dat.data[:] = self.own_output_coords.dat.data_ro[:]
            self.monitor.update_monitor()
            self.m.dat.data[:] = self.monitor.m.dat.data_ro[:]
        else:
            # self.mesh.coordinates.assign(self.x)
            self.own_output_coords.interpolate(Constant(self.R)*self.x)
            self.x_new.dat.data[:] = self.own_output_coords.dat.data_ro[:]
            self.monitor.get_monitor_on_new_mesh(self.monitor.m, self.x_old, self.x_new)
            self.m.dat.data[:] = self.monitor.m.dat.data_ro[:]

        self.mesh.coordinates.assign(self.xi)
        theta_new = assemble(self.thetaform)/self.total_area
        self.theta.assign(theta_new)

    def fakemonitor(self, snes, it, rnorm):
        cursol = snes.getSolution()
        self.update_mxtheta(cursol)  # updates m, x, and theta

    def get_first_mesh(self, initialise_fn):

        """
        This function is used to generate a mesh adapted to the initial state.
        :arg initialise_fn: a user-specified Python function that sets the
        initial condition
        """
        self.initial_mesh = True
        self.initialise_fn = initialise_fn
        self.mesh_solv.solve()

        # remake mesh solver with new tolerance
        self.params["snes_rtol"] = self.tol
        # self.params["snes_linesearch_type"] = "bt"
        self.params["snes_max_it"] = 15

        self.mesh_solv = NonlinearVariationalSolver(self.mesh_prob,
                                                    nullspace=self.nullspace,
                                                    transpose_nullspace=self.nullspace,
                                                    pre_jacobian_callback=self.update_mxtheta,
                                                    pre_function_callback=self.update_mxtheta,
                                                    solver_parameters=self.params)

        self.mesh_solv.snes.setMonitor(self.fakemonitor)
        self.initial_mesh = False

    def get_new_mesh(self):
        # Back up the current mesh
        self.x_old.dat.data[:] = self.mesh_in.coordinates.dat.data_ro[:]

        # Make monitor function
        # TODO: should I just pass in the 'coords to use' to update_monitor?
        self.monitor.mesh.coordinates.dat.data[:] = self.mesh_in.coordinates.dat.data_ro[:]
        self.monitor.update_monitor()

        # Back this up
        self.monitor.m_old.assign(self.monitor.m)

        # Generate new mesh, coords put in self.x
        try:
            self.mesh_solv.solve()
        except:
            print("mesh solver did not converge - oh well, continuing anyway!")

        # Move data from internal mesh to output mesh.
        self.own_output_coords.interpolate(Constant(self.R)*self.x)
        # self.mesh.coordinates.assign(self.x)
        # self.output_coords.dat.data[:] = self.mesh.coordinates.dat.data_ro[:]
        self.output_coords.dat.data[:] = self.own_output_coords.dat.data_ro[:]
        return self.output_coords

    def pre_meshgen_callback(self):
        if self.pre_meshgen_fn:
            self.pre_meshgen_fn()

    def post_meshgen_callback(self):
        if self.post_meshgen_fn:
            self.post_meshgen_fn()
