from abc import ABCMeta
from pyop2.profiling import timed_stage
from gusto.advection import Advection
from gusto.configuration import logger
from gusto.forcing import Forcing
from gusto.linear_solvers import LinearTimesteppingSolver
from gusto.state import FieldCreator
from firedrake import DirichletBC, Function


__all__ = ["Timestepper", "PrescribedAdvectionTimestepper", "CrankNicolson"]


class TimeLevelFields(FieldCreator):

    def __init__(self, *equations):
        super().__init__()
        for eqn in equations:
            name = eqn.field_name
            space = eqn.function_space
            subfield_names = eqn.fields if hasattr(eqn, "fields") else []
            self.create_field(name, *subfield_names, space=space)


class Timestepper(object, metaclass=ABCMeta):
    """
    Basic timestepping class for Gusto.

    :arg state: a :class:`.State` object
    :arg equations_schemes: a list of tuples (equation, schemes)
         pairing a :class:`.PrognosticEquation` object with the scheme(s)
         to use to timestep it. The schemes entry can be a :class:`.Advection`
         object which is applied to the entire equation, or a tuple of
         (scheme, *active_labels) where scheme is a :class:`.Advection`
         object and the active_labels are :class:`.Label` objects that
         specify which parts of the equation to apply this scheme to.
    :arg physics_list: optional list of classes that implement `physics` schemes
    :arg prescribed_fields: an ordered list of tuples, pairing a field name
         with a function that returns the field as a function of time.
    """

    def __init__(self, state, schemes,
                 physics_list=None, prescribed_fields=None):

        self.state = state

        if isinstance(schemes, Advection):
            self.schemes = tuple((schemes,))
        elif schemes is None:
            self.schemes = []
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

        self._setup_timeloop_fields()

    def _setup_timeloop_fields(self):
        """
        Sets up fields to contain the values at time levels n and n+1,
        getting the field's name and function space from the equation
        """
        self.xn = TimeLevelFields(*[scheme.equation for scheme in self.schemes])
        self.xnp1 = TimeLevelFields(*[scheme.equation for scheme in self.schemes])

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.xnp1("u")

        bcs = self.state.bcs

        for bc in bcs:
            bc.apply(unp1)

    def setup_schemes(self, schemes):
        """
        Setup the timestepping schemes
        """
        # if the child class has defined an advecting velocity, use it
        try:
            uadv = self.advecting_velocity
        except AttributeError:
            uadv = None

        for scheme in schemes:
            scheme.replace_advecting_velocity(uadv)

    def setup_timeloop(self, state, t, tmax, pickup):
        """
        Setup the timeloop by setting up timestepping schemes and diagnostics,
        dumping the fields and picking up from a previous run, if required
        """
        if pickup:
            t = state.pickup_from_checkpoint()
        self.setup_schemes(self.schemes)
        self.state.setup_diagnostics()
        with timed_stage("Dump output"):
            self.state.setup_dump(t, tmax, pickup)
        return t

    def initialise(self, state):
        """
        Set the fields in xn to their initial values from state.fields and
        evaluate all prescribed fields
        """
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
        for scheme in self.schemes:
            field_name = scheme.field_name
            scheme.apply(self.xn(field_name), self.xnp1(field_name))
            self.xn(field_name).assign(self.xnp1(field_name))

    def run(self, t, tmax, pickup=False):
        """
        This is the timeloop.
        """
        state = self.state
        dt = state.dt

        # initialise the fields in xn
        self.initialise(state)

        t = self.setup_timeloop(state, t, tmax, pickup)

        while t < tmax - 0.5*dt:
            logger.info("at start of timestep, t=%s, dt=%s" % (t, dt))

            t += dt
            state.t.assign(t)

            self.evaluate_prescribed_fields(state)

            # steps fields from xn to xnp1
            self.timestep(state)

            self.update_fields(self.xn, self.xnp1)
            self.update_fields(state.fields, self.xnp1)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            with timed_stage("Dump output"):
                state.dump(t)

        if state.output.checkpoint:
            state.chkpt.close()

        logger.info("TIMELOOP complete. t=%s, tmax=%s" % (t, tmax))


class PrescribedAdvectionTimestepper(Timestepper):
    """
    Timestepping class for solving equations with a prescribed advecting
    velocity

    :arg state: a :class:`.State` object
    :arg equations_schemes: a list of tuples (equation, schemes)
         pairing a :class:`.PrognosticEquation` object with the scheme(s)
         to use to timestep it. The schemes entry can be a :class:`.Advection`
         object which is applied to the entire equation, or a tuple of
         (scheme, *active_labels) where scheme is a :class:`.Advection`
         object and the active_labels are :class:`.Label` objects that
         specify which parts of the equation to apply this scheme to.
    :arg physics_list: optional list of classes that implement `physics` schemes
    :arg prescribed_fields: an ordered list of tuples, pairing a field name
         with a function that returns the field as a function of time.
    """

    def __init__(self, state, schemes,
                 physics_list=None, prescribed_fields=None):

        super().__init__(state,
                         schemes,
                         physics_list=physics_list,
                         prescribed_fields=prescribed_fields)

    @property
    def advecting_velocity(self):
        try:
            return self.state.fields("u")
        except AttributeError:
            raise ValueError("You have not specified an advecting velocity to use")


class CrankNicolson(Timestepper):
    """
    This class implements a Crank-Nicolson discretisation, with Strang
    splitting and auxilliary semi-Lagrangian advection.
    Defines the advecting velocity to be the average of u_n and u_{n+1}.
    After applying the semi implicit step passively advected fields are
    advected and any diffusion terms are applied.

    :arg state: a :class:`.State` object
    :arg equation_set: (optional) a :class:`.PrognosticMixedEquation` object,
    the main set of equations to be solved (eg ShallowWaterEquations)
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffused_fields: (optional) iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffuse, and the
        :class:`~.Diffusion` to use.
    :arg equations_schemes: a list of tuples (equation, schemes)
         pairing a :class:`.PrognosticEquation` object with the scheme(s)
         to use to timestep it. The schemes entry can be a :class:`.Advection`
         object which is applied to the entire equation, or a tuple of
         (scheme, *active_labels) where scheme is a :class:`.Advection`
         object and the active_labels are :class:`.Label` objects that
         specify which parts of the equation to apply this scheme to.
    :arg physics_list: optional list of classes that implement `physics` schemes
    :arg prescribed_fields: an order list of tuples, pairing a field name with a
         function that returns the field as a function of time.
    :kwargs: maxk is the number of outer iterations, maxi is the number of inner
             iterations and alpha is the offcentering parameter
    """

    def __init__(self, state, equation_set, schemes=None,
                 physics_list=None, prescribed_fields=None, **kwargs):

        self.maxk = kwargs.pop("maxk", 4)
        self.maxi = kwargs.pop("maxi", 1)
        self.alpha = kwargs.pop("alpha", 0.5)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        self.equation_set = equation_set

        self.active_advection = [scheme for scheme in schemes
                                 if scheme.field_name in equation_set.fields]

        # list of fields that are part of the semi implicit step but
        # are not advected (for example, when solving equations
        # linearised about a state of rest the velocity advection term
        # disappears)
        self.non_advected_fields = [
            name for name in
            set(equation_set.fields).difference(
                set([scheme.field_name for scheme in self.active_advection]))
        ]

        super().__init__(state,
                         schemes,
                         physics_list=physics_list,
                         prescribed_fields=prescribed_fields)

    @property
    def advecting_velocity(self):
        """
        This defines the advecting velocity as the average of the velocity
        at time level n and that at time level n+1.
        """
        un = self.xn("u")
        unp1 = self.xnp1("u")
        return un + self.alpha*(unp1-un)

    def _setup_timeloop_fields(self):
        """
        Sets up fields to contain the values at time levels n and n+1, as
        well as intermediate fields xstar (the results of applying the
        first forcing step) and xp (the result of applying the
        advection schemes). We also need functions to hold the
        solution of the linear system and the residual.

        """
        self.xn = TimeLevelFields(
            *[self.equation_set, *[scheme.equation for scheme in self.schemes]])
        self.xnp1 = TimeLevelFields(
            *[self.equation_set, *[scheme.equation for scheme in self.schemes]])
        self.xstar = TimeLevelFields(self.equation_set)
        self.xp = TimeLevelFields(self.equation_set)

        self.xrhs = Function(self.equation_set.function_space)
        self.dy = Function(self.equation_set.function_space)

    def setup_timeloop(self, state, t, tmax, pickup):
        """
        Setup the timeloop by setting up timestepping schemes and diagnostics,
        dumping the fields and picking up from a previous run, if required.
        We also need to set up the linear solver and the forcing solvers.
        """
        t = super().setup_timeloop(state, t, tmax, pickup)

        self.schemes = [scheme for scheme in self.schemes if scheme not in self.active_advection]
        if hasattr(self.equation_set, "linear_solver"):
            self.linear_solver = self.equation_set.linear_solver
        else:
            self.linear_solver = LinearTimesteppingSolver(self.equation_set,
                                                          state.dt, self.alpha)

        self.forcing = Forcing(self.equation_set, state.dt, self.alpha)
        return t

    def timestep(self, state):
        """
        This defines the timestep for the CrankNicolson scheme. First the
        semi_implicit_step timesteps the equation_set, then any additional
        equations are solved and finally diffusion is applied.
        """
        self.semi_implicit_step()

        for scheme in self.schemes:
            field_name = scheme.field_name
            scheme.apply(self.xn(field_name), self.xnp1(field_name))
            self.xn(field_name).assign(self.xnp1(field_name))

    def semi_implicit_step(self):
        """
        The semi implicit step for the CrankNicholson scheme. This applies
        the forcing terms (defined as those that are not labelled
        advection) for alpha*dt, then applies the advection schemes to
        the advection terms and finally applies the forcing terms for
        (1-alpha)*dt.
        """

        fname = self.equation_set.field_name
        with timed_stage("Apply forcing terms"):
            self.forcing.apply(self.xn(fname), self.xn(fname),
                               self.xstar(fname), label="explicit")

        for k in range(self.maxk):

            with timed_stage("Advection"):
                for scheme in self.active_advection:
                    field_name = scheme.field_name
                    # advects a field from xstar and puts result in xp
                    scheme.apply(self.xstar(field_name), self.xp(field_name))
                for name in self.non_advected_fields:
                    self.xp(name).assign(self.xstar(name))

            self.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(self.xp(fname), self.xnp1(fname),
                                       self.xrhs, label="implicit")

                self.xrhs -= self.xnp1(fname)

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve(self.xrhs, self.dy)  # solves linear system and places result in self.dy

                xnp1 = self.xnp1(fname)
                xnp1 += self.dy

            self._apply_bcs()
