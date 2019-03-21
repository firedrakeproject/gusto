from abc import ABCMeta, abstractproperty
from firedrake import (Function, TestFunction, inner, dx, div, cross, jump,
                       SpatialCoordinate, sqrt, FunctionSpace, avg, dS_v,
                       MixedFunctionSpace, TestFunctions, TrialFunctions,
                       FacetNormal, dot, grad)
from gusto.form_manipulation_labelling import (all_terms, advecting_velocity,
                                               subject, time_derivative,
                                               linearisation,
                                               drop, index, advection,
                                               replace_labelled,
                                               has_labels, Term, LabelledForm)
from gusto.diffusion import interior_penalty_diffusion_form
from gusto.linear_solvers import CompressibleSolver
from gusto.thermodynamics import pi as Pi
from gusto.transport_equation import (vector_invariant_form,
                                      continuity_form, advection_form,
                                      linear_continuity_form,
                                      advection_equation_circulation_form,
                                      advection_vector_manifold_form,
                                      kinetic_energy_form)
from gusto.state import build_spaces
from ufl import replace


class PrognosticEquation(object, metaclass=ABCMeta):
    """
    Base class for prognostic equations

    :arg state: :class:`.State` object
    :arg function space: :class:`.FunctionSpace` object, the function
         space that the equation is defined on
    :arg field_names: name(s) of the prognostic field(s)

    The class sets up the fields in state and registers them with the
    diagnostics class. It defines a mass term, labelled with the
    time_derivative label. All remaining forms must be defined in the
    child class form method. Calling this class returns the form
    mass_term + form
    """
    def __init__(self, state, function_space, field_name):

        self.state = state
        self.function_space = function_space
        self.field_name = field_name

        # default is to dump the field unless user has specified
        # otherwise when setting up the output parameters
        dump = state.output.dumplist or True
        state.fields(field_name, space=function_space, dump=dump, pickup=True)

        state.diagnostics.register(field_name)

    def __add__(self, other):
        if type(other) is LabelledForm:
            new = self.__class__(self.state, self.function_space, self.field_name)
            new.residual = self.residual + other
            return new


class AdvectionEquation(PrognosticEquation):
    """
    Class defining the advection equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object, the function
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the advection_form
    """
    def __init__(self, state, function_space, field_name,
                 **kwargs):
        super().__init__(state, function_space, field_name)

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = subject(time_derivative(inner(q, test)*dx), q)

        self.residual = (
            mass_form + advection_form(state, function_space, **kwargs)
        )


class ContinuityEquation(PrognosticEquation):
    """
    Class defining the continuity equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object, the function
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the continuity_form
    """

    def __init__(self, state, function_space, field_name,
                 **kwargs):
        super().__init__(state, function_space, field_name)

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = subject(time_derivative(inner(q, test)*dx), q)

        self.residual = (
            mass_form + continuity_form(state, function_space, **kwargs)
        )


class DiffusionEquation(PrognosticEquation):
    """
    Class defining the diffusion equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object, the function
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the diffuson_form
    """

    def __init__(self, state, function_space, field_name, **kwargs):

        super().__init__(state, function_space, field_name)

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = subject(time_derivative(inner(q, test)*dx), q)

        self.residual = (
            mass_form
            + interior_penalty_diffusion_form(state, function_space, **kwargs)
        )


class AdvectionDiffusionEquation(PrognosticEquation):
    """
    Class defining the advection-diffusion equation.

    :arg state: :class:`.State` object
    :arg field_name: name of the prognostic field
    :arg function_space: :class:`.FunctionSpace` object, the function
    :kwargs: any kwargs to be passed on to the advection_form or diffusion_form
    """
    def __init__(self, state, function_space, field_name, **kwargs):
        super().__init__(state, function_space, field_name)
        dkwargs = {}
        for k in ["kappa", "mu"]:
            assert k in kwargs.keys(), "diffusion form requires %s kwarg " % k
            dkwargs[k] = kwargs.pop(k)
        akwargs = kwargs

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = subject(time_derivative(inner(q, test)*dx), q)

        self.residual = (
            mass_form
            + advection_form(state, function_space, **akwargs)
            + interior_penalty_diffusion_form(state, function_space, **dkwargs)
        )


class PrognosticMixedEquation(PrognosticEquation):
    """
    Base class for the equation set defined on a mixed function space.
    Child classes must define their fields and solver parameters for
    the mixed system.

    :arg state: :class:`.State` object
    :arg function space: :class:`.FunctionSpace` object, the function
         space that the equations are defined on - this should be a
         mixed function space

    The class sets up the fields in state and registers them with the
    diagnostics class. It defines a mass term, labelled with the
    time_derivative label. All remaining forms must be defined in the
    child class form method. Calling this class returns the form
    mass_term + form
    """

    def __init__(self, state, function_space):

        self.state = state
        self.function_space = function_space

        assert len(function_space) == len(self.fields), "size of function space and number of fields should match"

        # default is to dump all fields unless user has specified
        # otherwise when setting up the output parameters
        dump = state.output.dumplist or self.fields
        state.fields(self.field_name, *self.fields, space=function_space,
                     dump=dump, pickup=True)

        state.diagnostics.register(*self.fields)

    @abstractproperty
    def field_name(self):
        """
        Child classes must define a name to use to access the mixed
        prognostic fields
        """
        pass

    @abstractproperty
    def fields(self):
        """
        Child classes must define a list of their prognostic field names.
        """
        pass

    @abstractproperty
    def solver_parameters(self):
        """
        Child classes must define default solver parameters for the
        mixed system.
        """
        pass


class ShallowWaterEquations(PrognosticMixedEquation):
    """
    Class defining the shallow water equations.

    :arg state: :class:`.State` object
    :arg family: str, specifies the velocity space family to use
    :arg degree: int, specifies the degree for the depth space
    :kwargs: (optional) expressions for additional fields and discretisation
    options to be passed to the form

    Default behaviour:
    * velocity advection term in vector invariant form.
    * Coriolis term present and the Coriolis parameter takes the value for
    the Earth. Pass in fexpr=None for non-rotating shallow water.
    """
    field_name = "sw"

    fields = ['u', 'D']

    solver_parameters = {
        'ksp_type': 'preonly',
        'mat_type': 'matfree',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {'ksp_type': 'cg',
                          'pc_type': 'gamg',
                          'ksp_rtol': 1e-8,
                          'mg_levels': {'ksp_type': 'chebyshev',
                                        'ksp_max_it': 2,
                                        'pc_type': 'bjacobi',
                                        'sub_pc_type': 'ilu'}}
    }

    def __init__(self, state, family, degree, **kwargs):

        fexpr = kwargs.pop("fexpr", "default")
        bexpr = kwargs.pop("bexpr", None)
        u_advection_option = kwargs.pop("u_advection_option", "vector_invariant_form")
        linear = kwargs.pop("linear", False)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        # define the function spaces
        Vu, VD = build_spaces(state, family, degree)
        W = MixedFunctionSpace((Vu, VD))

        super().__init__(state, W)

        g = state.parameters.g
        H = state.parameters.H

        w, phi = TestFunctions(W)
        trials = TrialFunctions(W)
        X = Function(W)
        u, D = X.split()

        # define mass term
        mass_form = (
            index(subject(time_derivative(inner(u, w)*dx), u), 0)
            + index(subject(time_derivative(inner(D, phi)*dx), D), 1)
        )

        # define velocity advection term
        if u_advection_option == "vector_invariant_form":
            u_adv = vector_invariant_form(state, W, 0)
        elif u_advection_option == "vector_advection":
            u_adv = advection_vector_manifold_form(state, W, 0)
        elif u_advection_option == "circulation_form":
            ke_form = kinetic_energy_form(state, W, 0)
            ke_form = advection.remove(ke_form)
            ke_form = ke_form.label_map(all_terms,
                                        replace_labelled(subject,
                                                         advecting_velocity))
            u_adv = advection_equation_circulation_form(state, W, 0) + ke_form
        else:
            raise ValueError("Invalid u_advection_option: %s" % u_advection_option)

        # define pressure gradient term and its linearisation
        pressure_gradient_term = subject(-g*div(w)*D*dx, X)
        linear_pg_term = pressure_gradient_term.label_map(
            all_terms, replace_labelled(trials, subject))

        # the base form for u contains the velocity advection term and
        # the pressure gradient term
        u_form = u_adv + linearisation(pressure_gradient_term, linear_pg_term)

        # setup optional coriolis and topography terms, default is for
        # the Coriolis term to be that for the Earth.
        if fexpr:
            if fexpr == "default":
                Omega = state.parameters.Omega
                x = SpatialCoordinate(state.mesh)
                R = sqrt(inner(x, x))
                fexpr = 2*Omega*x[2]/R

            V = FunctionSpace(state.mesh, "CG", 1)
            f = state.fields("coriolis", space=V)
            f.interpolate(fexpr)

            # define the coriolis term and its linearisation
            coriolis_term = subject(f*inner(w, state.perp(u))*dx, X)
            linear_coriolis_term = coriolis_term.label_map(
                all_terms, replace_labelled(trials, subject))
            # add on the coriolis term
            u_form += linearisation(coriolis_term, linear_coriolis_term)

        if bexpr:
            b = state.fields("topography", space=state.fields("D").function_space())
            b.interpolate(bexpr)
            # add on the topography term - the linearisation
            # is not defined as we don't usually make it part
            # of the linear solver, However, this will have to
            # be defined when we start using exponential
            # integrators.
            u_form += -g*div(w)*b*dx

        # define the depth continuity term and its linearisation
        Dadv = continuity_form(state, W, 1)
        Dadv_linear = linear_continuity_form(state, W, 1, qbar=H).label_map(
            all_terms, replace_labelled(trials, subject, advecting_velocity))
        D_form = linearisation(Dadv, Dadv_linear)

        self.residual = mass_form + index(u_form, 0) + index(D_form, 1)

        if linear:
            # grab the linearisation of each term (a bilinear form) and
            # apply to the term's subject to get the linear form
            def linearise_term(t):
                t_lin = t.get(linearisation)

                def get_action(tl):
                    subj = tl.get(subject).split()
                    new_form = replace(tl.form, {trials[0]: subj[0],
                                                 trials[1]: subj[1]})
                    t.labels.update(tl.labels)
                    return Term(new_form, t.labels)

                return t_lin.label_map(all_terms, get_action)

            linear_form = self.residual.label_map(
                has_labels(linearisation),
                linearise_term,
                drop)
            self.residual = mass_form + linear_form


class CompressibleEulerEquations(PrognosticMixedEquation):

    field_name = "compressible"

    fields = ['u', 'rho', 'theta']

    solver_parameters = {
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'ksp_type': 'gmres',
        'ksp_max_it': 100,
        'ksp_gmres_restart': 50,
        'pc_fieldsplit_schur_fact_type': 'FULL',
        'pc_fieldsplit_schur_precondition': 'selfp',
        'fieldsplit_0': {'ksp_type': 'preonly',
                         'pc_type': 'bjacobi',
                         'sub_pc_type': 'ilu'},
        'fieldsplit_1': {'ksp_type': 'preonly',
                         'pc_type': 'gamg',
                         'mg_levels': {'ksp_type': 'chebyshev',
                                       'ksp_chebyshev_esteig': True,
                                       'ksp_max_it': 1,
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}}
    }

    def __init__(self, state, family, horizontal_degree, vertical_degree,
                 **kwargs):

        u_advection_option = kwargs.pop("u_advection_option", "vector_invariant_form")
        Omega = kwargs.pop("Omega", None)
        linear = kwargs.pop("linear", False)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        # define the function spaces
        Vu, Vr, Vth = build_spaces(state, family, horizontal_degree,
                                   vertical_degree)
        W = MixedFunctionSpace((Vu, Vr, Vth))

        super().__init__(state, W)

        g = state.parameters.g
        cp = state.parameters.cp

        w, phi, gamma = TestFunctions(W)
        trials = TrialFunctions(W)
        X = Function(W)
        u, rho, theta = X.split()
        rhobar = state.fields("rhobar", space=Vr)
        thetabar = state.fields("thetabar", space=Vth)
        pi = Pi(self.state.parameters, rho, theta)
        n = FacetNormal(state.mesh)

        # define mass term
        mass_form = (
            index(subject(time_derivative(inner(u, w)*dx), u), 0)
            + index(subject(time_derivative(inner(rho, phi)*dx), rho), 1)
            + index(subject(time_derivative(inner(theta, gamma)*dx), theta), 2)
        )

        # define velocity advection term
        if u_advection_option == "vector_invariant_form":
            u_adv = vector_invariant_form(state, W, 0)
        elif u_advection_option == "vector_advection":
            u_adv = advection_vector_manifold_form(state, W, 0)
        elif u_advection_option == "circulation_form":
            ke_form = kinetic_energy_form(state, W, 0)
            ke_form = advection.remove(ke_form)
            ke_form = ke_form.label_map(all_terms,
                                        replace_labelled(subject,
                                                         advecting_velocity))
            u_adv = advection_equation_circulation_form(state, W, 0) + ke_form
        else:
            raise ValueError("Invalid u_advection_option: %s" % u_advection_option)

        # define pressure gradient term and its linearisation
        pressure_gradient_term = subject(
            cp*(-div(theta*w)*pi*dx + jump(theta*w, n)*avg(pi)*dS_v), u
        )

        # define gravity term and its linearisation
        gravity_term = Term(g*inner(state.k, w)*dx)

        # the base form for u contains the velocity advection term,
        # the pressure gradient term and the gravity term
        u_form = u_adv + pressure_gradient_term + gravity_term

        # define the coriolis term and its linearisation
        if Omega is not None:
            coriolis_term = subject(inner(cross(2*Omega, u), w)*dx, X)
            # add on the coriolis term
            u_form += coriolis_term

        # define the density continuity term and its linearisation
        rho_adv = continuity_form(state, W, 1)

        # define the  potential temperatur advection term and its linearisation
        theta_adv = advection_form(state, W, 2)

        self.residual = (
            mass_form + index(u_form, 0) + index(rho_adv, 1)
            + index(theta_adv, 2)
        )

        self.linear_solver = CompressibleSolver(state, self)
