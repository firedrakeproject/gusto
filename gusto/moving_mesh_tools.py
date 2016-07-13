from firedrake import Function, action, assemble, TestFunction, TrialFunction, dx, solve


class MovingMeshAdvection(object):

    def __init__(self, state, advection_dict, mesh_velocity_expr=None):
        self.state = state
        self.dt = state.timestepping.dt
        self.advection_dict = advection_dict
        self.mesh_velocity_expr = mesh_velocity_expr
        if mesh_velocity_expr is not None:
            self.mesh_velocity = Function(state.V[0]).project(mesh_velocity_expr)
            state.field_dict['mesh_velocity'] = self.mesh_velocity
        self.oldx = Function(state.mesh.coordinates.function_space())
        self.deltax = Function(state.mesh.coordinates.function_space())
        self.x = self.state.mesh.coordinates
        self.xa_fields = {}
        for name, func in state.field_dict.iteritems():
            self.xa_fields[name] = Function(func.function_space())

    def _get_mesh_velocity(self):
        if self.mesh_velocity_expr is not None:
            if hasattr(self.mesh_velocity_expr, "t"):
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

    def move_mesh(self):

        # assemble required mass matrices on old mesh
        mass_matrices = {}
        rhs = {}
        for field, advection in self.advection_dict.iteritems():
            if hasattr(advection, 'continuity') and advection.continuity:
                fs = self.state.field_dict[field].function_space()
                test = TestFunction(fs)
                trial = TrialFunction(fs)
                mass = test*trial*dx
                mass_matrices[field] = mass
                rhs[field] = assemble(action(mass, self.xa_fields[field]))
        # Move mesh
        deltax = self._get_deltax()
        self.x += deltax
        self.mesh_velocity.project(self.mesh_velocity_expr)

        for field, advection in self.advection_dict.iteritems():
            if hasattr(advection, 'continuity') and advection.continuity:
                lhs = assemble(mass_matrices[field])
                solve(lhs, self.xa_fields[field], rhs[field])

    def advection(self, xn_fields, xnp1_fields, t, k=None):
        first_time = (k == 0)
        if first_time:
            self.oldx.assign(self.x)
        elif k is not None:
            self.x.assign(self.oldx)
        self.t = t
        self._set_ubar()  # computes state.ubar from state.xn and state.xnp1
        for field, advection in self.advection_dict.iteritems():
            # advects a field from xn and puts result in xa
            advection.apply(xn_fields[field], self.xa_fields[field])

        self.move_mesh()

        # Second advection step on new mesh
        self._project_ubar()
        for field, advection in self.advection_dict.iteritems():
            # advects a field from xa and puts result in xnp1
            advection.apply(self.xa_fields[field], xnp1_fields[field])
