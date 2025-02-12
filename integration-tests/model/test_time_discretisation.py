from firedrake import norm
from gusto import *
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize(
    "scheme", [
        "ssprk2_increment_2", "ssprk2_predictor_2", "ssprk2_linear_2",
        "ssprk2_increment_3", "ssprk2_predictor_3", "ssprk2_linear_3",
        "ssprk2_increment_4", "ssprk2_predictor_4", "ssprk2_linear_4",

        "ssprk3_increment_3", "ssprk3_predictor_3", "ssprk3_linear_3",
        "ssprk3_increment_4", "ssprk3_predictor_4", "ssprk3_linear_4",
        "ssprk3_increment_5", "ssprk3_predictor_5", "ssprk3_linear_5",

        "ssprk4_increment_5", "ssprk4_predictor_5", "ssprk4_linear_5",

        "TrapeziumRule", "ImplicitMidpoint", "QinZhang_increment", "QinZhang_predictor",
        "RK4", "Heun", "BDF2", "TR_BDF2", "AdamsBashforth", "Leapfrog",
        "AdamsMoulton", "AdamsMoulton"
    ]
)
def test_time_discretisation(tmpdir, scheme, tracer_setup):
    if (scheme == "AdamsBashforth"):
        # Tighter stability constraints
        geometry = "sphere"
        setup = tracer_setup(tmpdir, geometry, small_dt=True)
        domain = setup.domain
        V = domain.spaces("DG")
        eqn = AdvectionEquation(domain, V, "f")
    else:
        geometry = "sphere"
        setup = tracer_setup(tmpdir, geometry)
        domain = setup.domain
        V = domain.spaces("DG")
        eqn = AdvectionEquation(domain, V, "f")

    if scheme == "ssprk2_increment_2":
        transport_scheme = SSPRK2(domain, rk_formulation=RungeKuttaFormulation.increment)
    elif scheme == "ssprk2_predictor_2":
        transport_scheme = SSPRK2(domain, rk_formulation=RungeKuttaFormulation.predictor)
    elif scheme == "ssprk2_linear_2":
        transport_scheme = SSPRK2(domain, rk_formulation=RungeKuttaFormulation.linear)
    elif scheme == "ssprk2_increment_3":
        transport_scheme = SSPRK2(domain, rk_formulation=RungeKuttaFormulation.increment, stages=3)
    elif scheme == "ssprk2_predictor_3":
        transport_scheme = SSPRK2(domain, rk_formulation=RungeKuttaFormulation.predictor, stages=3)
    elif scheme == "ssprk2_linear_3":
        transport_scheme = SSPRK2(domain, rk_formulation=RungeKuttaFormulation.linear, stages=3)
    elif scheme == "ssprk2_increment_4":
        transport_scheme = SSPRK2(domain, rk_formulation=RungeKuttaFormulation.increment, stages=4)
    elif scheme == "ssprk2_predictor_4":
        transport_scheme = SSPRK2(domain, rk_formulation=RungeKuttaFormulation.predictor, stages=4)
    elif scheme == "ssprk2_linear_4":
        transport_scheme = SSPRK2(domain, rk_formulation=RungeKuttaFormulation.linear, stages=4)

    elif scheme == "ssprk3_increment_3":
        transport_scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.increment)
    elif scheme == "ssprk3_predictor_3":
        transport_scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.predictor)
    elif scheme == "ssprk3_linear_3":
        transport_scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.linear)
    elif scheme == "ssprk3_increment_4":
        transport_scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.increment, stages=4)
    elif scheme == "ssprk3_predictor_4":
        transport_scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.predictor, stages=4)
    elif scheme == "ssprk3_linear_4":
        transport_scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.linear, stages=4)

    elif scheme == "ssprk3_increment_5":
        transport_scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.increment, stages=5)
    elif scheme == "ssprk3_predictor_5":
        transport_scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.predictor, stages=5)
    elif scheme == "ssprk3_linear_5":
        transport_scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.linear, stages=5)

    elif scheme == "ssprk4_increment_5":
        transport_scheme = SSPRK4(domain, rk_formulation=RungeKuttaFormulation.increment)
    elif scheme == "ssprk4_predictor_5":
        transport_scheme = SSPRK4(domain, rk_formulation=RungeKuttaFormulation.predictor)
    elif scheme == "ssprk4_linear_5":
        transport_scheme = SSPRK4(domain, rk_formulation=RungeKuttaFormulation.linear)

    elif scheme == "TrapeziumRule":
        transport_scheme = TrapeziumRule(domain)
    elif scheme == "ImplicitMidpoint":
        transport_scheme = ImplicitMidpoint(domain)
    elif scheme == "QinZhang_increment":
        transport_scheme = QinZhang(domain, rk_formulation=RungeKuttaFormulation.increment)
    elif scheme == "QinZhang_predictor":
        transport_scheme = QinZhang(domain, rk_formulation=RungeKuttaFormulation.predictor)
    elif scheme == "RK4":
        transport_scheme = RK4(domain)
    elif scheme == "Heun":
        transport_scheme = Heun(domain)
    elif scheme == "BDF2":
        transport_scheme = BDF2(domain)
    elif scheme == "TR_BDF2":
        transport_scheme = TR_BDF2(domain, gamma=0.5)
    elif scheme == "Leapfrog":
        # Leapfrog unstable with DG
        domain.spaces.create_space("CG1", "CG", 1)
        Vf = domain.spaces("CG1")
        eqn = AdvectionEquation(domain, Vf, "f")
        transport_scheme = Leapfrog(domain)
    elif scheme == "AdamsBashforth":
        transport_scheme = AdamsBashforth(domain, order=2)
    elif scheme == "AdamsMoulton":
        transport_scheme = AdamsMoulton(domain, order=2)

    transport_method = DGUpwind(eqn, 'f')

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, transport_scheme, setup.io, time_varying_velocity, transport_method
    )

    # Initial conditions
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)
    assert run(timestepper, setup.tmax, setup.f_end) < setup.tol
