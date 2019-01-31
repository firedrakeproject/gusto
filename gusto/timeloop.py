from abc import ABCMeta, abstractmethod
from pyop2.profiling import timed_stage
from gusto.advection import Advection
from gusto.configuration import logger
from gusto.equations import PrognosticEquation
from gusto.forcing import Forcing
from gusto.form_manipulation_labelling import advection, diffusion
from gusto.linear_solvers import LinearTimesteppingSolver
from gusto.state import FieldCreator
from firedrake import DirichletBC, Function

__all__ = ["Timestepper", "PrescribedAdvectionTimestepper", "CrankNicolson"]


class Timestepper(object, metaclass=ABCMeta):
    """
    Basic timestepping class for Gusto.

    :arg state: a :class:`.State` object
    :arg equation_set: (optional) a :class:`.PrognosticEquation` object,
         defined on a mixed function space
    :arg equations: a list of tuples (field, equation)
         pairing a field name with the :class:`.PrognosticEquation` object
         that defines the prognostic equation that the field satisfies.
    :arg schemes: either a list of tuples (field, scheme, *active_labels)
         that specify which scheme to use to timestep the field's prognostic
         equation and which labels, if any, to use to select part of the
         equation; or an :class:`.Advection` object which specifies the
         scheme to apply to the equation_set.
    :arg physics_list: optional list of classes that implement `physics` schemes
    :arg prescribed_fields: an ordered list of tuples, pairing a field name
         with a function that returns the field as a function of time.
    """

    def __init__(self, state, equation_set=None, *,
                 equations=None, schemes=None,
                 physics_list=None, prescribed_fields=None):

        self.state = state

        if equations is not None:
            self.equations = tuple(equations)
        else:
            self.equations = []
        if schemes is None:
            raise ValueError("You need to specify which timestepping schemes you are using")
        elif isinstance(schemes, Advection):
            self.schemes = (('X', schemes),)
        else:
            self.schemes = tuple(schemes)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []
        if prescribed_fields is not None:
            self.prescribed_fields = prescribed_fields
        else:
            self.prescribed_fields = []

        self.xn = FieldCreator()
        self.xnp1 = FieldCreator()

        if equation_set is not None:
            assert isinstance(equation_set, PrognosticEquation)
            assert len(equation_set.function_space) > 1
            W = equation_set.function_space
            fieldlist = equation_set.fieldlist
            self.xn(*fieldlist, space=W, dump=False, pickup=False)
            self.xnp1(*fieldlist, space=W, dump=False, pickup=False)
            self.equations.insert(0, ('X', equation_set))

        for field, eqn in self.equations:
            self.xn(field, space=state.fields(field).function_space())
            self.xnp1(field, space=state.fields(field).function_space())

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.xnp1("u")

        if unp1.function_space().extruded:
            M = unp1.function_space()
            bcs = [DirichletBC(M, 0.0, "bottom"),
                   DirichletBC(M, 0.0, "top")]

            for bc in bcs:
                bc.apply(unp1)

    def setup_schemes(self):
        """
        Setup the timestepping schemes
        """
        state = self.state
        equations = self.equations
        # if the child class has defined an advecting velocity, use it
        try:
            uadv = self.advecting_velocity
        except AttributeError:
            uadv = None
        for name, scheme, *active_labels in self.schemes:
            field = state.fields(name)
            if name in dict(equations).keys():
                eqn = dict(equations)[name]
            else:
                eqn = dict(equations)['X']
                assert name in eqn.fieldlist
            scheme.setup(state, field, eqn, self.dt, *active_labels,
                         u_advecting=uadv)

    def setup_timeloop(self, t, dt, tmax, pickup):
        """
        Setup the timeloop by setting up timestepping schemes and diagnostics,
        dumping the fields and picking up from a previous run, if required
        """
        self.setup_schemes()
        self.state.setup_diagnostics(dt)
        with timed_stage("Dump output"):
            self.state.setup_dump(dt, tmax, pickup)
            t = self.state.dump(t, pickup)
        return t

    def initialise(self, state):
        for field in self.xn:
            field.assign(state.fields(field.name()))
        self.evaluate_prescribed_fields(state)

    def evaluate_prescribed_fields(self, state):
        for name, evaluation in self.prescribed_fields:
            state.fields(name).project(evaluation(state.t))

    def update_fields(self, old, new):
        for field in new:
            old(field.name()).assign(field)

    def timestep(self, state):
        for name, scheme, *active_labels in self.schemes:
            scheme.apply(self.xn(name), self.xnp1(name))
            self.xn(name).assign(self.xnp1(name))

    def run(self, t, dt, tmax, pickup=False):
        """
        This is the timeloop.
        """
        state = self.state
        self.dt = dt

        self.initialise(state)

        t = self.setup_timeloop(t, dt, tmax, pickup)

        while t < tmax - 0.5*dt:
            logger.info("at start of timestep, t=%s, dt=%s" % (t, dt))

            t += dt
            state.t.assign(t)

            self.evaluate_prescribed_fields(state)

            # steps fields from xn to xnp1
            self.timestep(state)

            # update xn
            self.update_fields(self.xn, self.xnp1)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            self.update_fields(self.xn, self.xnp1)
            self.update_fields(state.fields, self.xnp1)

            with timed_stage("Dump output"):
                state.dump(t, pickup=False)

        if state.output.checkpoint:
            state.chkpt.close()

        logger.info("TIMELOOP complete. t=%s, tmax=%s" % (t, tmax))


class PrescribedAdvectionTimestepper(Timestepper):
    """
    Timestepping class for solving equations with a prescribed advecting
    velocity

    :arg state: a :class:`.State` object
    :arg equations: a list of tuples (field, equation)
         pairing a field name with the :class:`.PrognosticEquation` object
         that defines the prognostic equation that the field satisfies.
    :arg schemes: either a list of tuples (field, scheme, *active_labels)
         that specify which scheme to use to timestep the field's prognostic
         equation and which labels, if any, to use to select part of the
         equation; or an :class:`.Advection` object which specifies the
         scheme to apply to the equation_set.
    :arg physics_list: optional list of classes that implement `physics` schemes
    :arg prescribed_fields: an ordered list of tuples, pairing a field name
         with a function that returns the field as a function of time.
    """

    def __init__(self, state, *,
                 equations=None, schemes=None,
                 physics_list=None, prescribed_fields=None):

        super().__init__(state,
                         equations=equations,
                         schemes=schemes,
                         physics_list=physics_list,
                         prescribed_fields=prescribed_fields)

    @property
    def advecting_velocity(self):
        try:
            return self.state.fields("u")
        except AttributeError:
            raise ValueError("You have not specified an advecting velocity to use")


class SemiImplicitTimestepper(Timestepper):
    """
    Base class for semi implicit schemes.
    Defines the advecting velocity to be the average of u_n and u_{n+1}.
    After applying the semi implicit step (which is defined in the child
    class), passively advected fields are advected and any diffusion terms
    are applied.
    """

    def __init__(self, state, equation_set, *, equations, schemes=None,
                 physics_list=None, prescribed_fields=None):

        super().__init__(state, equation_set,
                         equations=equations,
                         schemes=schemes,
                         physics_list=physics_list,
                         prescribed_fields=prescribed_fields)

        self.xstar = FieldCreator()
        self.xstar(*equation_set.fieldlist, space=equation_set.function_space)

        self.passive_advection = [(name, scheme, *active_labels)
                                  for name, scheme, *active_labels in schemes
                                  if name not in equation_set.fieldlist]

    @property
    def advecting_velocity(self):
        un = self.xn("u")
        unp1 = self.xnp1("u")
        return un + self.alpha*(unp1-un)

    @abstractmethod
    def semi_implicit_step(self):
        """
        Implement the semi implicit step for the timestepping scheme.
        """
        pass

    def timestep(self, state):

        self.semi_implicit_step()

        for name, scheme in self.passive_advection:
            # advects a field from xn and puts result in xnp1
            scheme.apply(self.xn(name), self.xnp1(name))

        with timed_stage("Diffusion"):
            for name, scheme in self.diffused_fields:
                scheme.apply(self.xnp1(name), self.xnp1(name))


class CrankNicolson(SemiImplicitTimestepper):
    """
    This class implements a Crank-Nicolson discretisation, with Strang
    splitting and auxilliary semi-Lagrangian advection.

    :arg state: a :class:`.State` object
    :arg equation_set: (optional) a :class:`.PrognosticEquation` object,
         defined on a mixed function space
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffuse, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    :arg prescribed_fields: an order list of tuples, pairing a field name with a
         function that returns the field as a function of time.
    :kwargs: maxk is the number of outer iterations, maxi is the number of inner
             iterations and alpha is the offcentering parameter
    """

    def __init__(self, state, equation_set, *, equations=None,
                 advected_fields=None, diffused_fields=None,
                 physics_list=None, prescribed_fields=None, **kwargs):

        self.maxk = kwargs.pop("maxk", 4)
        self.maxi = kwargs.pop("maxi", 1)
        self.alpha = kwargs.pop("alpha", 0.5)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        # list of (field, scheme) for fields that are advected as part
        # of the semi implicit step
        self.active_advection = [
            (name, scheme)
            for name, scheme in advected_fields
            if name in equation_set.fieldlist]

        # list of fields that are part of the semi implicit step but
        # are not advected (for example, when solving equations
        # linearised about a state of rest the velocity advection term
        # disappears)
        self.non_advected_fields = [
            name for name in
            set(equation_set.fieldlist).difference(
                set(dict(advected_fields).keys()))
        ]

        # create a list of (field, scheme, *active_labels) from the
        # advected_fields and diffused_fields that have been passed in
        schemes = []
        for field, scheme in advected_fields:
            schemes.append((field, scheme, advection))
        if diffused_fields is None:
            assert all([not t.has_label(diffusion) for t in equation_set()]),\
                "you have some diffusion terms but have not specified which scheme to use to apply them"
            self.diffused_fields = []
        else:
            self.diffused_fields = diffused_fields
            for field, scheme in diffused_fields:
                schemes.append((field, scheme, diffusion))

        super().__init__(state, equation_set,
                         equations=equations,
                         schemes=schemes,
                         physics_list=physics_list,
                         prescribed_fields=prescribed_fields)

        self.equation_set = equation_set
        W = equation_set.function_space
        self.xp = FieldCreator()
        self.xp(*equation_set.fieldlist, space=W)

        self.xrhs = Function(W)
        self.dy = Function(W)

    def setup_timeloop(self, t, dt, tmax, pickup):
        t = super().setup_timeloop(t, dt, tmax, pickup)
        self.linear_solver = LinearTimesteppingSolver(self.equation_set,
                                                      dt, self.alpha)
        self.forcing = Forcing(self.equation_set, dt, self.alpha)
        return t

    def semi_implicit_step(self):

        with timed_stage("Apply forcing terms"):
            self.forcing.apply(self.xn.X, self.xn.X,
                               self.xstar.X, label="explicit")

        for k in range(self.maxk):

            with timed_stage("Advection"):
                for name, scheme in self.active_advection:
                    # advects a field from xstar and puts result in xp
                    scheme.apply(self.xstar(name), self.xp(name))
                for name in self.non_advected_fields:
                    self.xp(name).assign(self.xstar(name))

            self.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(self.xp.X, self.xnp1.X,
                                       self.xrhs, label="implicit")

                self.xrhs -= self.xnp1.X

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve(self.xrhs, self.dy)  # solves linear system and places result in self.dy

                self.xnp1.X += self.dy

            self._apply_bcs()
