from firedrake import as_vector, exp, norm
from gusto import *


def run(tracer_setup):

    setup = tracer_setup(geometry="slice", blob=True)
    state = setup.state
    dt = setup.dt
    tmax = setup.tmax
    f_init = setup.f_init

    fspace = state.spaces("DG")
    state.fields("f_exact", space=fspace)

    x = SpatialCoordinate(state.mesh)
    L = 10.

    u = state.fields("u", space=state.spaces("HDiv"))
    u.project(as_vector([10., 0.]))

    def f_exact(t):
        d = 5.
        xs = x[0] - 0.5*L
        return (1/(1+4*t))*((exp(-d*(2*xs+d)) + exp(d*(2*xs-d)))*f_init)**(1/(1+4*t))

    equations = [
        ("f", AdvectionDiffusionEquation(state, fspace, "f", kappa=1., mu=5))]
    f = state.fields("f")
    f.interpolate(f_init)

    schemes = []
    schemes.append(("f", SSPRK3(), advection))
    schemes.append(("f", BackwardEuler(), diffusion))

    prescribed_fields = [("f_exact", f_exact)]
    timestepper = PrescribedAdvectionTimestepper(
        state, equations=equations, schemes=schemes,
        prescribed_fields=prescribed_fields)

    timestepper.run(0, dt=dt, tmax=tmax)
    return timestepper.state.fields("f_minus_f_exact")


def test_advection_diffusion(tracer_setup):

    err = run(tracer_setup)
    assert norm(err) < 5e-2
