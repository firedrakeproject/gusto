"""
A simple sanity-check example: solves the scalar transport (advection)
equation

    dq/dt + u . grad(q) = 0

for a scalar q in a DG0 space on a periodic 1D domain, with the transporting
wind u set to zero everywhere. Since there is no wind, q should not evolve at
all -- the field should remain (up to solver tolerances) equal to its initial
condition.

This uses the `PrescribedTransport` timestepper -- a `Timestepper` subclass
specialised for equations whose transporting wind is a prescribed (rather
than prognostic) field -- together with the SSPRK3 explicit Runge-Kutta time
discretisation.
"""

from firedrake import (PeriodicIntervalMesh, SpatialCoordinate, Constant,
                       as_vector, VectorFunctionSpace, norm)
from gusto import (Domain, IO, OutputParameters, AdvectionEquation,
                  DGUpwind, SSPRK3, PrescribedTransport)


def build_stepper(dirname, ncells=20, L=10.0, dt=0.05):
    mesh = PeriodicIntervalMesh(ncells, L)
    # "CG", 1 gives a DG0 space for the scalar transport variable
    domain = Domain(mesh, dt, "CG", 1)

    V = domain.spaces("DG")
    Vu = VectorFunctionSpace(mesh, "DG", 0)
    eqn = AdvectionEquation(domain, V, "q", Vu=Vu)

    output = OutputParameters(dirname=dirname, dumpfreq=100)
    io = IO(domain, output)

    transport_method = [DGUpwind(eqn, "q")]

    scheme = SSPRK3(domain)
    stepper = PrescribedTransport(
        eqn, scheme, io, False, transport_method
    )

    return stepper, mesh, V


def run_zero_wind_transport(dirname="results/zero_wind_transport"):
    dt = 0.05
    stepper, mesh, V = build_stepper(dirname, dt=dt)

    x, = SpatialCoordinate(mesh)
    q0_expr = 1.0 + 0.5*x

    q0 = stepper.fields("q")
    q0.interpolate(q0_expr)
    u0 = stepper.fields("u")
    u0.project(as_vector([Constant(0.0)]))

    initial_q = q0.copy(deepcopy=True)

    stepper.run(t=0, tmax=10*dt)

    final_q = stepper.fields("q")
    error = norm(final_q - initial_q) / norm(initial_q)

    print(f"Relative change in q with zero wind: {error:.3e}")
    assert error < 1e-12, "Field evolved despite zero transporting wind!"
    print("PASSED: field did not evolve, as expected with zero wind.")


if __name__ == "__main__":
    run_zero_wind_transport()
