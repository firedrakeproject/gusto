from abc import ABCMeta, abstractmethod
from pyop2.profiling import timed_stage
from gusto.configuration import logger
from gusto.forcing import Forcing
from gusto.form_manipulation_labelling import advection
from gusto.linear_solvers import LinearTimesteppingSolver
from gusto.state import FieldCreator
from firedrake import DirichletBC, Function

__all__ = ["Timestepper", "CrankNicolson"]


class Timestepper(object, metaclass=ABCMeta):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    :arg prescribed_fields: an order list of tuples, pairing a field name with a
         function that returns the field as a function of time.
    """

    def __init__(self, state, *, equations=None,
                 advected_fields=None, diffused_fields=None,
                 physics_list=None, prescribed_fields=None):

        self.state = state

        if advected_fields is None:
            self.advected_fields = ()
        else:
            self.advected_fields = tuple(advected_fields)
        if diffused_fields is None:
            self.diffused_fields = ()
        else:
            self.diffused_fields = tuple(diffused_fields)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []
        if prescribed_fields is not None:
            self.prescribed_fields = prescribed_fields
        else:
            self.prescribed_fields = []

        state.xn = FieldCreator()
        state.xnp1 = FieldCreator()

        if equations is not None:
            self.fieldlist = equations.fieldlist
            state.xn(self.fieldlist, state.spaces.W, dump=False, pickup=False)
            state.xnp1(self.fieldlist, state.spaces.W, dump=False, pickup=False)
        else:
            self.fieldlist = []

        additional_fields = set([f for f, _ in self.advected_fields + self.diffused_fields]).difference(set(self.fieldlist))
        for field in additional_fields:
            state.xn(field, state.fields(field).function_space())
            state.xnp1(field, state.fields(field).function_space())

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.state.xnp1("u")

        if unp1.function_space().extruded:
            M = unp1.function_space()
            bcs = [DirichletBC(M, 0.0, "bottom"),
                   DirichletBC(M, 0.0, "top")]

            for bc in bcs:
                bc.apply(unp1)

    def setup_timeloop(self, t, tmax, pickup):
        """
        Setup the timeloop by setting up diagnostics, dumping the fields and
        picking up from a previous run, if required
        """
        for name, scheme in self.advected_fields:
            scheme.setup(self.state, u_advecting="prescribed")

        self.state.setup_diagnostics()
        with timed_stage("Dump output"):
            self.state.setup_dump(tmax, pickup)
            t = self.state.dump(t, pickup)
        return t

    def timestep(self):

        state = self.state

        for name, scheme in self.advected_fields:
            old_field = getattr(state.xn, name)
            new_field = getattr(state.xnp1, name)
            scheme.apply(old_field, new_field)

        for field in state.xnp1:
            state.xn(field.name()).assign(field)
            state.fields(field.name()).assign(field)

    def run(self, t, tmax, pickup=False):
        """
        This is the timeloop. After completing the semi implicit step
        any passively advected fields are updated, implicit diffusion and
        physics updates are applied (if required).
        """

        state = self.state
        dt = state.timestepping.dt

        t = self.setup_timeloop(t, tmax, pickup)

        for field in state.xn:
            field.assign(state.fields(field.name()))

        while t < tmax - 0.5*dt:
            logger.info("at start of timestep, t=%s, dt=%s" % (t, dt))

            t += dt
            state.t.assign(t)

            for field in state.xnp1:
                field.assign(state.xn(field.name()))

            for name, evaluation in self.prescribed_fields:
                state.fields(name).project(evaluation(t))

            self.timestep()

            for field in state.xn:
                field.assign(state.xnp1(field.name()))

            with timed_stage("Diffusion"):
                for name, diffusion in self.diffused_fields:
                    field = state.xnp1(field.name())
                    diffusion.apply(field, field)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            for field in state.xnp1:
                state.xn(field.name()).assign(field)
                state.fields(field.name()).assign(field)

            with timed_stage("Dump output"):
                state.dump(t, pickup=False)

        if state.output.checkpoint:
            state.chkpt.close()

        logger.info("TIMELOOP complete. t=%s, tmax=%s" % (t, tmax))


class SemiImplicitTimestepper(Timestepper):

    def __init__(self, state, *, equations, advected_fields,
                 diffused_fields=None, physics_list=None, prescribed_fields=None):

        super().__init__(state, equations=equations,
                         advected_fields=advected_fields,
                         diffused_fields=diffused_fields,
                         physics_list=physics_list,
                         prescribed_fields=prescribed_fields)

        self.xstar = FieldCreator()
        self.xstar(self.fieldlist, state.spaces("W"))

    @property
    def advecting_velocity(self):
        un = self.state.xn("u")
        unp1 = self.state.xnp1("u")
        return un + self.alpha*(unp1-un)

    @property
    def passive_advection(self):
        """
        Advected fields that are not part of the semi implicit step are
        passively advected
        """
        return [(name, scheme)
                for name, scheme in self.advected_fields
                if name not in self.fieldlist]

    @abstractmethod
    def semi_implicit_step(self):
        """
        Implement the semi implicit step for the timestepping scheme.
        """
        pass

    def timestep(self):

        state = self.state

        self.semi_implicit_step()

        for name, scheme in self.passive_advection:
            old_field = getattr(state.xn, name)
            new_field = getattr(state.xnp1, name)
            # advects a field from xn and puts result in xnp1
            scheme.update_ubar(self.advecting_velocity)
            scheme.apply(old_field, new_field)


class CrankNicolson(SemiImplicitTimestepper):
    """
    This class implements a Crank-Nicolson discretisation, with Strang
    splitting and auxilliary semi-Lagrangian advection.

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    :arg prescribed_fields: an order list of tuples, pairing a field name with a
         function that returns the field as a function of time.
    """

    def __init__(self, state, *, equations, advected_fields,
                 diffused_fields=None, physics_list=None,
                 prescribed_fields=None, **kwargs):

        self.maxk = kwargs.pop("maxk", 4)
        self.maxi = kwargs.pop("maxi", 1)
        self.alpha = kwargs.pop("alpha", 0.5)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))
        self.dt = state.timestepping.dt

        super().__init__(state, equations=equations,
                         advected_fields=advected_fields,
                         diffused_fields=diffused_fields,
                         physics_list=physics_list,
                         prescribed_fields=prescribed_fields)

        self.linear_solver = LinearTimesteppingSolver(state, equations, self.dt, self.alpha)

        self.forcing = Forcing(state, equations, self.dt, self.alpha)

        self.xp = FieldCreator()
        self.xp(self.fieldlist, state.spaces("W"))

        self.xrhs = Function(state.spaces("W"))
        self.dy = Function(state.spaces("W"))

        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [
            (name, scheme)
            for name, scheme in advected_fields
            if name in self.fieldlist]

        for name, scheme in self.active_advection:
            scheme.setup(state, labels=[advection])

        for name, scheme in self.passive_advection:
            scheme.setup(state, advecting_velocity="prescribed")

        self.non_advected_fields = [name for name in set(self.fieldlist).difference(set(dict(advected_fields).keys()))]

    def semi_implicit_step(self):
        state = self.state

        with timed_stage("Apply forcing terms"):
            self.forcing.apply(state.xn.X, state.xn.X,
                               self.xstar.X, label="explicit")

        for k in range(self.maxk):

            with timed_stage("Advection"):
                # first computes ubar from state.xn and state.xnp1
                for name, scheme in self.active_advection:
                    scheme.update_ubar(self.advecting_velocity)
                    # advects a field from xstar and puts result in xp
                    scheme.apply(self.xstar(name), self.xp(name))
                for name in self.non_advected_fields:
                    self.xp(name).assign(self.xstar(name))

            self.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(self.xp.X, state.xnp1.X,
                                       self.xrhs, label="implicit")

                self.xrhs -= state.xnp1.X

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve(self.xrhs, self.dy)  # solves linear system and places result in self.dy

                state.xnp1.X += self.dy

            self._apply_bcs()
