from firedrake import Function, action, assemble, TestFunction, TrialFunction, dx, solve, inner


class MovingMeshAdvection(object):

    def __init__(self, state, advection_dict, meshx_callback, meshv_callback, uadv=None, uexpr=None):
        self.state = state
        self.dt = state.timestepping.dt
        self.advection_dict = advection_dict
        self.meshx = meshx_callback
        self.meshv = meshv_callback

        self.oldx = Function(state.mesh.coordinates.function_space())
        self.deltax = Function(state.mesh.coordinates.function_space())
        self.x = self.state.mesh.coordinates
        self.mesh_velocity = Function(state.mesh.coordinates.function_space())
        state.field_dict['mesh_velocity'] = self.mesh_velocity

        self.xa_fields = {}
        for name, func in state.field_dict.iteritems():
            self.xa_fields[name] = Function(func.function_space())
        self.uadv = uadv
        self.uexpr = uexpr
        self.ubar = Function(state.V[0])

    def _set_ubar(self):
        """
        Update ubar in the advection methods.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]
        self.uadv.project(self.uexpr)
        self.meshv(self)
        if self.uadv is not None:
            self.ubar.project(self.uadv - self.mesh_velocity)
        else:
            self.ubar.assign(un + state.timestepping.alpha*(unp1-un) - self.mesh_velocity)

        for field, advection in self.advection_dict.iteritems():
            advection.ubar.assign(self.ubar)

    def _project_ubar(self):
        """
        Update ubar in the advection methods after mesh has moved.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]
        self.uadv.project(self.uexpr)
        self.meshv(self)
        if self.uadv is not None:
            self.ubar.project(self.uadv - self.mesh_velocity)
        else:
            self.ubar.assign(un + state.timestepping.alpha*(unp1-un) - self.mesh_velocity)

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
        self.meshx(self)

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
