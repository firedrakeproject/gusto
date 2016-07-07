from firedrake import Function, action, assemble, TestFunction, TrialFunction, dx, solve


class MovingMeshAdvection(object):

    def __init__(self, state, advection_dict, mesh_velocity_expr=None):
        self.state = state
        self.dt = state.timestepping.dt
        self.advection_dict = advection_dict
        self.mesh_velocity_expr = mesh_velocity_expr
        if mesh_velocity_expr is not None:
            self.mesh_velocity = Function(state.V[0]).project(mesh_velocity_expr)
        self.deltax = Function(state.mesh.coordinates.function_space())

    def _get_mesh_velocity(self):
        if self.mesh_velocity_expr is not None:
            self.mesh_velocity_expr.t = self.t
            return self.mesh_velocity.project(self.mesh_velocity_expr)

    def _get_deltax(self):
        return self.deltax.project(self.dt*self.mesh_velocity)

    def _set_ubar(self):
        """
        Update ubar in the advection methods.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]
        v = self._get_mesh_velocity()

        for field, advection in self.advection_dict.iteritems():
            advection.ubar.assign(un + state.timestepping.alpha*(unp1-un) - v)

    def _project_ubar(self):
        """
        Update ubar in the advection methods after mesh has moved.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]
        v = self._get_mesh_velocity()

        for field, advection in self.advection_dict.iteritems():
            advection.ubar.project(un + state.timestepping.alpha*(unp1-un) - v)

    def move_mesh(self, xstar_fields):

        # assemble required mass matrices on old mesh
        mass_matrices = {}
        rhs = {}
        for field, advection in self.advection_dict.iteritems():
            if advection.continuity:
                fs = self.state.field_dict[field].function_space()
                test = TestFunction(fs)
                trial = TrialFunction(fs)
                mass = test*trial*dx
                mass_matrices[field] = mass
                rhs[field] = assemble(action(mass, xstar_fields[field]))

        # Move mesh
        x = self.state.mesh.coordinates
        deltax = self._get_deltax()
        x += deltax
        self.mesh_velocity.project(self.mesh_velocity_expr)

        for field, advection in self.advection_dict.iteritems():
            if advection.continuity:
                lhs = assemble(mass_matrices[field])
                solve(lhs, xstar_fields[field], rhs[field])

    def advection(self, xn_fields, xstar_fields, xnp1_fields, t):
        self.t = t
        self._set_ubar()  # computes state.ubar from state.xn and state.xnp1
        for field, advection in self.advection_dict.iteritems():
            # advects a field from xn and puts result in xstar
            advection.apply(xn_fields[field], xstar_fields[field])

        self.move_mesh(xstar_fields)

        # Second advection step on new mesh
        self._project_ubar()
        for field, advection in self.advection_dict.iteritems():
            # advects a field from xstar and puts result in xnp1
            advection.apply(xstar_fields[field], xnp1_fields[field])
