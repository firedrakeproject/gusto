from firedrake import Function, action, assemble, TestFunction, TrialFunction, dx, solve, inner


class MovingMeshAdvection(object):

    def __init__(self, state, advection_dict, mesh_velocity_expr=None, uadv=None, uexpr=None):
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
        self.uadv = uadv
        self.uexpr = uexpr
        self.ubar = Function(state.V[0])

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
        self.uadv.interpolate(self.uexpr)
        v = self._get_mesh_velocity()
        if self.uadv is not None:
            self.ubar.project(self.uadv - v)
        else:
            self.ubar.assign(un + state.timestepping.alpha*(unp1-un) - v)

        for field, advection in self.advection_dict.iteritems():
            advection.ubar.assign(self.ubar)

    def _project_ubar(self):
        """
        Update ubar in the advection methods after mesh has moved.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]
        self.uadv.interpolate(self.uexpr)
        v = self._get_mesh_velocity()
        if self.uadv is not None:
            self.ubar.project(self.uadv - v)
        else:
            self.ubar.assign(un + state.timestepping.alpha*(unp1-un) - v)

        for field, advection in self.advection_dict.iteritems():
            advection.ubar.project(self.ubar)

    def _setup_move_mesh(self):
        # setup required mass matrices on old mesh
        self.mass_matrices = {}
        self.rhs = {}
        for field, advection in self.advection_dict.iteritems():
            if hasattr(advection, 'continuity') and advection.continuity or field is "u":
                fs = self.state.field_dict[field].function_space()
                test = TestFunction(fs)
                trial = TrialFunction(fs)
                mass = inner(test,trial)*dx
                self.mass_matrices[field] = mass
                self.rhs[field] = Function(fs)

    def move_mesh(self, k):

        if k == 0 or k is None:
            self._setup_move_mesh()

        # assemble required mass matrices on old mesh
        for field, advection in self.advection_dict.iteritems():
            if hasattr(advection, 'continuity') and advection.continuity or field is "u":
                mass = self.mass_matrices[field]
                self.rhs[field].assign(assemble(action(mass, self.xa_fields[field])))

        # Move mesh
        deltax = self._get_deltax()
        self.x += deltax
        self.mesh_velocity.project(self.mesh_velocity_expr)

        for field, advection in self.advection_dict.iteritems():
            if hasattr(advection, 'continuity') and advection.continuity or field is "u":
                lhs = assemble(self.mass_matrices[field])
                solve(lhs, self.xa_fields[field], self.rhs[field])

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
            advection.apply(xn_fields[field], self.xa_fields[field], scale=0.5)

        self.move_mesh(k)

        # Second advection step on new mesh
        self._project_ubar()
        for field, advection in self.advection_dict.iteritems():
            # advects a field from xa and puts result in xnp1
            advection.apply(self.xa_fields[field], xnp1_fields[field], scale=0.5)
