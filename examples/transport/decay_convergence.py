"""
Tests that idealised physics parametrisations are correctly evaluated within
the different formulations of the explicit Runge-Kutta time discretisations
in Gusto.

The test problem is the scalar transport equation

    dq/dt + u . grad(q) = -k*q

for a scalar q in a DG0 space on a periodic 1D domain, with the transporting
wind u set to zero everywhere (so the only thing that can make q evolve is
the physics parametrisation). The right-hand side "-k*q" is supplied by the
`LinearDecay` physics parametrisation (gusto/physics/linear_decay.py), which
was written as a minimal test case: it simply evaluates k*q as a source term.

Since the wind is zero, the exact solution is the simple exponential decay

    q(t) = q(0) * exp(-k*t)

By running each Runge-Kutta scheme/formulation for a small, fixed number of
timesteps at a sequence of decreasing dt, and comparing against this analytic
solution, we can check:

  1) that each scheme converges at (at least) its expected order of accuracy;
  2) that the physics source term is being incorporated consistently across
     the different `RungeKuttaFormulation` options (increment, predictor and
     linear), since a bug in how physics is coupled into a particular stage
     would show up either as an incorrect answer or as a reduced/incorrect
     convergence rate.
"""

import numpy as np
from firedrake import (PeriodicIntervalMesh, VectorFunctionSpace, Constant,
                       as_vector, norm)
from gusto import (Domain, IO, OutputParameters, AdvectionEquation,
                  DGUpwind, PrescribedTransport, LinearDecay,
                  RungeKuttaFormulation, ForwardEuler, Heun, RK4,
                  SSPRK2, SSPRK3)


# Physical/test parameters ---------------------------------------------------
k = 2.0          # decay rate in dq/dt = -k*q
q0_value = 1.0   # initial (uniform) value of q
nsteps = 5       # small, fixed number of timesteps used for each run
L = 10.0         # domain length
ncells = 5       # coarse mesh: spatial error is irrelevant, since q is uniform


def analytic_solution(t):
    """Exact solution of dq/dt = -k*q, with q(0) = q0_value."""
    return q0_value * np.exp(-k*t)


def run_decay(dt, scheme_class, rk_formulation, **scheme_kwargs):
    """
    Runs the transport equation (with zero wind) plus the LinearDecay physics
    parametrisation, for `nsteps` timesteps of size `dt`, using the given
    Runge-Kutta `scheme_class` and `rk_formulation`.

    Returns:
        float: the numerical value of q at the final time.
    """

    mesh = PeriodicIntervalMesh(ncells, L)
    # "CG", 1 gives a DG0 space for the scalar transport variable
    domain = Domain(mesh, dt, "CG", 1)

    V = domain.spaces("DG")
    Vu = VectorFunctionSpace(mesh, "DG", 0)
    eqn = AdvectionEquation(domain, V, "q", Vu=Vu)

    physics_parametrisations = [LinearDecay(eqn, "q", k)]

    output = OutputParameters(dirname="results/decay_convergence", dumpfreq=100,
                              overwrite_files=True)
    io = IO(domain, output)

    transport_method = [DGUpwind(eqn, "q")]

    scheme = scheme_class(domain, rk_formulation=rk_formulation, **scheme_kwargs)

    stepper = PrescribedTransport(
        eqn, scheme, io, False, transport_method,
        physics_parametrisations=physics_parametrisations
    )

    q0 = stepper.fields("q")
    q0.interpolate(Constant(q0_value))
    u0 = stepper.fields("u")
    u0.project(as_vector([Constant(0.0)]))

    tmax = nsteps*dt
    stepper.run(t=0, tmax=tmax)

    final_q = stepper.fields("q")
    # q should remain spatially uniform -- take the norm as a proxy for its
    # (constant) value, normalised by the volume of the domain
    return norm(final_q) / np.sqrt(L)


def convergence_rate(scheme_class, rk_formulation, expected_order,
                     dts=(0.05, 0.025, 0.0125), **scheme_kwargs):
    """
    Computes the numerical solution for a sequence of timestep sizes, and
    estimates the convergence rate by comparing the errors (relative to the
    analytic solution) for consecutive values of dt.
    """
    errors = []
    for dt in dts:
        tmax = nsteps*dt
        q_numerical = run_decay(dt, scheme_class, rk_formulation, **scheme_kwargs)
        q_exact = analytic_solution(tmax)
        errors.append(abs(q_numerical - q_exact))

    rates = [
        np.log(errors[i]/errors[i+1]) / np.log(dts[i]/dts[i+1])
        for i in range(len(dts)-1)
    ]

    return errors, rates


def test_all_schemes():
    schemes = [
        ("ForwardEuler", ForwardEuler, 1, {}),
        ("Heun", Heun, 2, {}),
        ("SSPRK2 (2-stage)", SSPRK2, 2, {}),
        ("SSPRK3", SSPRK3, 3, {}),
        ("RK4", RK4, 4, {}),
    ]
    formulations = [
        RungeKuttaFormulation.increment,
        RungeKuttaFormulation.predictor,
        RungeKuttaFormulation.linear,
    ]

    print(f"{'Scheme':<20}{'Formulation':<14}{'Errors (coarse->fine)':<40}{'Rates':<30}")
    print("-"*104)

    results = {}
    for name, scheme_class, expected_order, kwargs in schemes:
        for rk_formulation in formulations:
            errors, rates = convergence_rate(
                scheme_class, rk_formulation, expected_order, **kwargs
            )
            errors_str = ", ".join(f"{e:.2e}" for e in errors)
            rates_str = ", ".join(f"{r:.2f}" for r in rates)
            print(f"{name:<20}{rk_formulation.name:<14}{errors_str:<40}{rates_str:<30}")

            # Allow some tolerance below the expected order, since we are
            # only using a handful of timesteps
            min_acceptable_rate = expected_order - 0.3
            passed = all(r > min_acceptable_rate for r in rates)
            results[(name, rk_formulation)] = passed

    print("-"*104)

    failures = {k: v for k, v in results.items() if not v}
    if not failures:
        print("PASSED: all schemes/formulations converge at their expected order, "
             "confirming the physics source term is correctly evaluated within "
             "each Runge-Kutta formulation.")
        return

    failed_formulations = {rk.name for (_, rk) in failures}
    print("FAILED for the following (scheme, formulation) combinations:")
    for (name, rk_formulation) in failures:
        print(f"  - {name} / {rk_formulation.name}")

    if failed_formulations == {RungeKuttaFormulation.linear.name}:
        # Every failure is under the "linear" formulation
        print(
            "\nDiagnosis: only the 'linear' RungeKuttaFormulation fails to "
            "converge once a physics source term is present; 'increment' and "
            "'predictor' both converge correctly. This points to a bug in how "
            "gusto/time_discretisation/explicit_runge_kutta.py builds the "
            "residual for the 'linear' formulation: unlike 'increment'/"
            "'predictor', its `res` property does not special-case terms with "
            "the `source_label`, so `replace_subject` substitutes the "
            "Runge-Kutta stage combination directly in place of the physics "
            "scheme's own auxiliary 'source' field -- discarding the k*q value "
            "computed by `LinearDecay.evaluate()`. As a result, physics "
            "parametrisations are not currently evaluated correctly when using "
            "the 'linear' Runge-Kutta formulation."
        )

    raise RuntimeError(
        "One or more Runge-Kutta scheme/formulation combinations did not "
        "converge at the expected order -- this suggests an error in how "
        "the physics parametrisation is being evaluated within the scheme."
    )


if __name__ == "__main__":
    test_all_schemes()

