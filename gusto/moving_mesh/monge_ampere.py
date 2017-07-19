from __future__ import absolute_import
import numpy as np
from firedrake import Function, VectorFunctionSpace, SpatialCoordinate, Mesh, \
    FunctionSpace, TensorFunctionSpace, dx, assemble, Constant, split, \
    TrialFunctions, TestFunctions, sqrt, dot, grad, cos, sin, Identity, outer, \
    inner, div, det, TrialFunction, TestFunction, LinearVariationalProblem, \
    LinearVariationalSolver, solve, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, VectorSpaceBasis, MixedVectorSpaceBasis, \
    replace, par_loop, READ, WRITE
from gusto.moving_mesh.mesh_generator import MeshGenerator


class OptimalTransportMeshGenerator(MeshGenerator):
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
        if hasattr(mesh_in, '_radius'):
            self.R = mesh_in._radius
            self.Rc = Constant(self.R)
        else:
            raise RuntimeError("Mesh doesn't seem to be an IcosahedralSphereMesh or a CubedSphereMesh")

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

        # And a version that lives on the internal mesh
        self.own_output_coords = Function(VectorFunctionSpace(self.mesh, "Q" if quads else "P", mesh_in.coordinates.ufl_element().degree()))

        # If user is using a P2/Q2 mesh, we can assign instead of interpolate
        # later on
        self.samecoorddegree = (mesh_in.coordinates.ufl_element().degree() == 2)

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
        self.total_area = assemble(Constant(1.0)*dxdeg(domain=self.mesh))

        # Set up functions
        self.phisigma = Function(MixedSpace)
        self.phi, self.sigma = split(self.phisigma)
        self.phisigma_temp = Function(MixedSpace)
        self.phi_temp, self.sigma_temp = split(self.phisigma_temp)

        # mesh coordinates
        self.x = Function(self.mesh.coordinates)  # for 'physical' coords
        self.xi = Function(self.mesh.coordinates)  # for 'computational' coords

        # monitor function mesh coords
        self.x_old = Function(monitor.mesh.coordinates)
        self.x_new = Function(monitor.mesh.coordinates)

        self.m = Function(P1)
        self.theta = Constant(0.0)

        # Define mesh equations
        v, tau = TestFunctions(MixedSpace)

        # sphere
        modgphi = sqrt(dot(grad(self.phi), grad(self.phi)) + 1e-14)
        expxi = self.xi*cos(modgphi) + grad(self.phi)*sin(modgphi)/modgphi
        projxi = Identity(3) - outer(self.xi, self.xi)

        expxi_temp = replace(expxi, {self.phisigma: self.phisigma_temp})

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

        # TODO: change for plane
        self.gradphi_cts2 = Function(W_cts)  # extra, as gradphi_cts not necessarily tangential

        # Set up initial sigma
        sigma_ = TrialFunction(TensorP2)
        tau_ = TestFunction(TensorP2)
        sigma_ini = Function(TensorP2)

        asigmainit = inner(sigma_, tau_)*dxdeg

        # TODO: change for plane
        Lsigmainit = -dot(div(tau_), expxi)*dxdeg

        solve(asigmainit == Lsigmainit, sigma_ini, solver_parameters={'ksp_type': 'cg'})

        self.phisigma.sub(1).assign(sigma_ini)

        # Solver options for mesh generation
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
                       "fieldsplit_0_mg_levels_ksp_max_it": 5,
                       "fieldsplit_0_mg_levels_pc_type": "ilu",
                       "fieldsplit_1_pc_type": "ilu",
                       "fieldsplit_1_ksp_type": "preonly",
                       "ksp_max_it": 100,
                       "snes_max_it": 50,
                       "ksp_gmres_restart": 100,
                       "snes_rtol": initial_tol,
                       "snes_linesearch_type": "bt",
                       # "ksp_monitor": True,
                       # "snes_monitor": True,
                       # "snes_linesearch_monitor": True,
                       "snes_lag_preconditioner": -1}

        self.mesh_solv = NonlinearVariationalSolver(self.mesh_prob,
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

        # Generate coords from this. TODO: change for plane
        # On sphere, firstly "fix grad(phi)" by ensuring that
        # grad(phi).x = 0, assuming |x| = 1
        par_loop("""
for (int i=0; i<vnew.dofs; i++) {
    double dot = 0.0;
    for (int j=0; j<3; j++) {
        dot += x[i][j]*vold[i][j];
    }
    for (int j=0; j<3; j++) {
        vnew[i][j] = vold[i][j] - dot*x[i][j];
    }
}
""", dx, {'x': (self.mesh.coordinates, READ),
          'vold': (self.gradphi_cts, READ),
          'vnew': (self.gradphi_cts2, WRITE)})

        # Then use exponential map to obtain x
        par_loop("""
for (int i=0; i<xi.dofs; i++) {
    double norm = 0.0;
    for (int j=0; j<3; j++) {
        norm += u[i][j]*u[i][j];
    }

    norm = sqrt(norm);

    if(norm < 1.0e-14) {
        for (int j=0; j<3; j++) {
            xout[i][j] = xi[i][j];
        }
    } else {
        for (int j=0; j<3; j++) {
            xout[i][j] = xi[i][j]*cos(norm) + (u[i][j]/norm)*sin(norm);
        }
    }
}
""", dx, {'xi': (self.xi, READ),
          'u': (self.gradphi_cts2, READ),
          'xout': (self.x, WRITE)})

        if self.initial_mesh:
            # Make coords suitable for the input mesh, self.mesh_in
            if self.samecoorddegree:
                self.own_output_coords.assign(Constant(self.R)*self.x)
            else:
                self.mesh.coordinates.assign(self.x)
                self.own_output_coords.interpolate(SpatialCoordinate(self.mesh))
                self.own_output_coords.dat.data[:] *= (self.R / np.linalg.norm(self.own_output_coords.dat.data, axis=1)).reshape(-1, 1)

            # Set them (note: this modifies the user-mesh!)
            self.output_coords.dat.data[:] = self.own_output_coords.dat.data_ro[:]
            self.mesh_in.coordinates.assign(self.output_coords)

            # Call user function to set initial data
            self.initialise_fn()

            # Make monitor function on user's mesh
            self.monitor.mesh.coordinates.dat.data[:] = self.own_output_coords.dat.data_ro[:]
            self.monitor.update_monitor()

            # Copy this to internal mesh
            self.m.dat.data[:] = self.monitor.m.dat.data_ro[:]

        else:
            # "Copy" self.x over to self.x_new
            if self.samecoorddegree:
                self.own_output_coords.assign(Constant(self.R)*self.x)
            else:
                self.mesh.coordinates.assign(self.x)
                self.own_output_coords.interpolate(SpatialCoordinate(self.mesh))
                self.own_output_coords.dat.data[:] *= (self.R / np.linalg.norm(self.own_output_coords.dat.data, axis=1)).reshape(-1, 1)

            self.x_new.dat.data[:] = self.own_output_coords.dat.data_ro[:]

            # Update representation of monitor function
            self.monitor.get_monitor_on_new_mesh(self.monitor.m, self.x_old, self.x_new)

            # Copy into own representation of monitor function
            self.m.dat.data[:] = self.monitor.m.dat.data_ro[:]

        # Set mesh coordinates to use "computational mesh"
        self.mesh.coordinates.assign(self.xi)

        # Calculate theta
        self.theta.assign(assemble(self.thetaform)/self.total_area)

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
        self.params["snes_linesearch_type"] = "bt"
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
        self.mesh_solv.solve()

        # Move data from internal mesh to output mesh.
        if self.samecoorddegree:
            self.own_output_coords.assign(Constant(self.R)*self.x)
        else:
            self.mesh.coordinates.assign(self.x)
            self.own_output_coords.interpolate(SpatialCoordinate(self.mesh))
            self.own_output_coords.dat.data[:] *= (self.R / np.linalg.norm(self.own_output_coords.dat.data, axis=1)).reshape(-1, 1)

        self.output_coords.dat.data[:] = self.own_output_coords.dat.data_ro[:]
        return self.output_coords
