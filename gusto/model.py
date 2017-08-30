from firedrake import sqrt, inner, SpatialCoordinate, FunctionSpace, as_vector
from gusto.advection import SSPRK3, ThetaMethod
from gusto.configuration import CompressibleParameters
from gusto.forcing import ShallowWaterForcing, CompressibleForcing
from gusto.linear_solvers import ShallowWaterSolver, CompressibleSolver
from gusto.transport_equation import VectorInvariant, AdvectionEquation, SUPGAdvection, EulerPoincare


class Model(object):

    def __init__(self,
                 state,
                 physical_domain,
                 parameters,
                 timestepping,
                 linear_solver,
                 forcing,
                 advected_fields,
                 diffused_fields,
                 physics_list):

        self.state = state
        self.physical_domain = physical_domain
        self.parameters = parameters
        self.timestepping = timestepping
        self.linear_solver = linear_solver
        self.forcing = forcing
        self.advected_fields = advected_fields
        self.diffused_fields = diffused_fields
        self.physics_list = physics_list


def ShallowWaterModel(state,
                      physical_domain, *,
                      parameters,
                      timestepping,
                      is_rotating=True,
                      coriolis_parameter=None,
                      linear_solver=None,
                      forcing=None,
                      advected_fields=None,
                      diffused_fields=None,
                      physics_list=None):

    if is_rotating:
        physical_domain.is_rotating = True
        fs = FunctionSpace(physical_domain.mesh, "CG", 1)
        f = state.fields("coriolis", fs)
        if coriolis_parameter is None:
            if physical_domain.on_sphere:
                x = SpatialCoordinate(physical_domain.mesh)
                f.interpolate(2*parameters.Omega*x[2]/sqrt(inner(x, x)))
            else:
                raise ValueError("There is no default Coriolis parameter for non spherical shallow water simulations.")
        else:
            f.interpolate(coriolis_parameter)

    if linear_solver is None:
        beta = timestepping.dt*timestepping.alpha
        linear_solver = ShallowWaterSolver(state, parameters, beta)

    if advected_fields is None:
        advected_fields = []
    field_scheme = dict(advected_fields)
    if "D" not in field_scheme.keys():
        Deqn = AdvectionEquation(physical_domain, state.spaces("DG"),
                                 state.spaces("HDiv"),
                                 equation_form="continuity")
        advected_fields.append(("D", SSPRK3(state.fields("D"),
                                            timestepping.dt, Deqn)))
    if "u" not in field_scheme.keys():
        ueqn = VectorInvariant(physical_domain, state.spaces("HDiv"),
                               state.spaces("HDiv"))
        advected_fields.append(("u",
                                ThetaMethod(state.fields("u"),
                                            timestepping.dt, ueqn)))

    if forcing is None:
        field_scheme = dict(advected_fields)
        euler_poincare = isinstance(field_scheme["u"].equation, EulerPoincare)
        forcing = ShallowWaterForcing(state, parameters, physical_domain, euler_poincare=euler_poincare)

    return Model(state, physical_domain, parameters, timestepping, linear_solver, forcing, advected_fields, diffused_fields, physics_list)


def CompressibleEulerModel(state,
                           physical_domain, *,
                           is_rotating=True,
                           rotation_vector=None,
                           parameters=None,
                           timestepping=None,
                           linear_solver=None,
                           forcing=None,
                           advected_fields=None,
                           diffused_fields=None,
                           physics_list=None):

    if is_rotating:
        physical_domain.is_rotating = True
        if rotation_vector is None:
            physical_domain.rotation_vector = as_vector((0., 0., 0.5e-4))
        else:
            physical_domain.rotation_vector = rotation_vector

    if parameters is None:
        parameters = CompressibleParameters()

    if linear_solver is None:
        beta = timestepping.dt*timestepping.alpha
        linear_solver = CompressibleSolver(state, parameters, beta, physical_domain.vertical_normal)

    if advected_fields is None:
        advected_fields = []
    field_scheme = dict(advected_fields)
    if "rho" not in field_scheme.keys():
        rhoeqn = AdvectionEquation(physical_domain, state.spaces("DG"),
                                   state.spaces("HDiv"),
                                   equation_form="continuity")
        advected_fields.append(("rho",
                                SSPRK3(state.fields("rho"),
                                       timestepping.dt, rhoeqn)))
    if "theta" not in field_scheme.keys():
        thetaeqn = SUPGAdvection(physical_domain, state.spaces("HDiv_v"),
                                 state.spaces("HDiv"),
                                 dt=timestepping.dt,
                                 supg_params={"dg_direction": "horizontal"},
                                 equation_form="advective")
        advected_fields.append(("theta",
                                SSPRK3(state.fields("theta"),
                                       timestepping.dt, thetaeqn)))
    if "u" not in field_scheme.keys():
        ueqn = VectorInvariant(physical_domain, state.spaces("HDiv"),
                               state.spaces("HDiv"))
        advected_fields.append(("u",
                                ThetaMethod(state.fields("u"),
                                            timestepping.dt, ueqn)))

    if forcing is None:
        field_scheme = dict(advected_fields)
        euler_poincare = isinstance(field_scheme["u"], EulerPoincare)
        forcing = CompressibleForcing(state, parameters, physical_domain, euler_poincare=euler_poincare)

    return Model(state, physical_domain, parameters, timestepping, linear_solver, forcing, advected_fields, diffused_fields, physics_list)


def AdvectionDiffusionModel(state,
                            physical_domain,
                            timestepping=None,
                            advected_fields=None,
                            diffused_fields=None):

    if not(advected_fields or diffused_fields):
        raise ValueError("You must provide the list of tuples of advected and/or diffused fields and schemes.")

    return Model(state, physical_domain, parameters=None, timestepping=timestepping, linear_solver=None, forcing=None, advected_fields=advected_fields, diffused_fields=diffused_fields, physics_list=None)
