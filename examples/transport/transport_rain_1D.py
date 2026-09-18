"""
An idealised transport-physics test case.

Water vapour is advected at constant velocity around a periodic 1D domain,
and simultaneously subject to an instantaneous condensation ("rain")
parametrisation which removes any vapour above a spatially- and
(in general) time-varying saturation profile. Since the saturation profile
here is constant-in-time and simply advects with the flow (it is expressed in
terms of the moving coordinate), the "rain" produced is a genuinely
time-dependent process, providing a non-trivial coupled transport-physics
test case with an explicit analytic solution.

This script builds and runs the model for a given dt and choice of
timestepper, and reports the L2 error of the rain and water vapour fields
against the analytic solution. It also produces a plot, at the end of the
run, of the final rain field against the analytic rainfall profile.

Two timestepping strategies are available, selected with `--timestepper`:
    - 'runge_kutta': physics is called at every stage of an SSPRK3 scheme
       (via `PrescribedTransport`, physics embedded in the transport
       equation's spatial discretisation).
    - 'split': a Strang-split approach where a full SSPRK3 transport step is
       taken, followed by a separate physics update (via
       `SplitPrescribedTransport`). The physics update may be "superstepped",
       i.e. called only every `physics_frequency` timesteps, by passing
       `--physics_frequency`.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from firedrake import (
    PeriodicIntervalMesh, SpatialCoordinate, Function, VectorFunctionSpace,
    conditional, cos, sin, acos, pi, errornorm, as_vector
)

from gusto import (
    Domain, IO, OutputParameters, CoupledTransportEquation, ActiveTracer,
    TracerVariableType, TransportEquationType, DGUpwind, SSPRK3, ForwardEuler,
    PrescribedTransport, SplitPrescribedTransport, InstantRain
)

# ---------------------------------------------------------------------------
# Test case parameters
# ---------------------------------------------------------------------------
L = 1.0          # domain length
u0 = 1.0         # (constant) advecting velocity
ncells = 40      # number of mesh cells
degree = 1       # DG space degree
Csat = 1.0       # mean saturation value
Ksat = 0.5       # amplitude of saturation variation
C0 = 0.8         # mean initial vapour value
K0 = 0.3         # amplitude of initial vapour variation
tmax = L / u0    # simulation time (one full transit of the domain)


def saturation_expr(x):
    return Csat + Ksat*cos(2*pi*x[0]/L)


def vapour_init_expr(x):
    return C0 + K0*cos(2*pi*x[0]/L)


def vapour_analytic_expr(x):
    """
    Since the saturation minimum (Csat - Ksat) is less than the vapour
    minimum (C0 - K0), condensation eventually removes all the "excess"
    vapour above the saturation minimum, everywhere in the domain, leaving
    a spatially-uniform vapour field equal to (Csat - Ksat).
    """
    global_min_msat = Csat - Ksat
    m0 = vapour_init_expr(x)
    return conditional(m0 < global_min_msat, m0, global_min_msat)


# Analytic rainfall profile at t=tmax, derived using the level-set/rearrangement
# technique of Section 2 of the accompanying write-up (eqn 2.11), generalised
# to account for the full periodic transit of the domain (our test uses
# parameters for which Ksat > K0, so the initial vapour profile is already
# locally supersaturated, m0(x) > msat(x), over part of the domain -- this
# produces an additional immediate/local condensation contribution, on top of
# the "arriving parcels" contribution described by eqn 2.11, which the
# write-up's simpler examples did not need to consider).
#
# msat and m0 are both decreasing on x in [0, L/2] (from their maxima at x=0)
# and increasing on x in [L/2, L] (back up to their maxima at x=L).
#
# - On the decreasing branch [0, L/2], the rain at a point x_t is the sum of:
#   (a) an "arriving parcels" contribution: this is eqn 2.11 (with our x
#       coordinate directly playing the role of paper's "x - lx/2", since the
#       two are equivalent under the periodic cos/sin phase), but with the
#       parcel-origin threshold x0_max additionally clipped to x_t itself
#       (since only parcels with origin x0 <= x_t can have reached x_t on a
#       direct pass), and doubled to account for the equal contribution from
#       parcels wrapping around through the increasing branch;
#   (b) a local/instantaneous term, max(0, m0(x_t) - msat(x_t)), from
#       parcels which start already supersaturated exactly at x_t.
# - On the increasing branch (L/2, L), no parcel ever arrives with a lower
#   value than it started with (msat only increases along direct paths, and
#   any wrapped parcel has already been clipped down to the global minimum
#   before re-entering this branch), so only the local term (b) contributes.
def rain_analytic_expr(x):
    """
    UFL expression for the analytic rainfall profile at t=tmax.
    """
    xt = x[0]
    theta = 2*pi*xt/L
    m0_val = C0 + K0*cos(theta)
    msat_val = Csat + Ksat*cos(theta)

    self_term = conditional(m0_val > msat_val, m0_val - msat_val, 0.0)

    arg = (msat_val - C0) / K0
    arg_clipped = conditional(arg > 1.0, 1.0, conditional(arg < -1.0, -1.0, arg))
    x0_max = (L/(2*pi)) * acos(arg_clipped)
    width = conditional(
        arg >= 1.0, 0.0,
        conditional(arg <= -1.0, 2*xt, 2*conditional(xt < x0_max, xt, x0_max))
    )
    minus_dmsat_dx = (2*pi*Ksat/L) * sin(theta)
    arriving_term = width * minus_dmsat_dx

    rain_decreasing_branch = arriving_term + self_term

    return conditional(xt <= L/2, rain_decreasing_branch, self_term)


def rain_analytic_numpy(x):
    """
    Numpy equivalent of `rain_analytic_expr`, for plotting purposes.
    """
    x = np.asarray(x, dtype=float)
    theta = 2*np.pi*x/L
    m0_val = C0 + K0*np.cos(theta)
    msat_val = Csat + Ksat*np.cos(theta)

    self_term = np.maximum(0.0, m0_val - msat_val)

    arg = (msat_val - C0) / K0
    x0_max = (L/(2*np.pi)) * np.arccos(np.clip(arg, -1.0, 1.0))
    width = np.where(
        arg >= 1.0, 0.0,
        np.where(arg <= -1.0, 2*x, 2*np.minimum(x, x0_max))
    )
    minus_dmsat_dx = (2*np.pi*Ksat/L) * np.sin(theta)
    arriving_term = width * minus_dmsat_dx

    rain_decreasing_branch = arriving_term + self_term

    return np.where(x <= L/2, rain_decreasing_branch, self_term)


def build_stepper(dt, timestepper='split', physics_frequency=1, dirname=None):
    mesh = PeriodicIntervalMesh(ncells, L)
    # "CG", 1 gives a DG1 space for the transport variables on a 1D domain
    domain = Domain(mesh, dt, 'CG', degree)
    x = SpatialCoordinate(mesh)

    Vu = VectorFunctionSpace(mesh, 'DG', 0)

    water_vapour = ActiveTracer(
        name='water_vapour', space='DG', variable_type=TracerVariableType.mixing_ratio,
        transport_eqn=TransportEquationType.advective
    )
    rain = ActiveTracer(
        name='rain', space='DG', variable_type=TracerVariableType.mixing_ratio,
        transport_eqn=TransportEquationType.no_transport
    )

    eqn = CoupledTransportEquation(
        domain, active_tracers=[water_vapour, rain], Vu=Vu
    )

    transport_method = DGUpwind(eqn, 'water_vapour')

    if dirname is None:
        dirname = f'moisture_transport_convergence_{timestepper}_dt{dt}'
    output = OutputParameters(dirname=dirname, dumpfreq=1000)
    io = IO(domain, output)

    instant_rain = InstantRain(
        eqn, saturation_expr(x), vapour_name='water_vapour', rain_name='rain',
        gamma_r=1.0, tau=None, convective_feedback=False
    )

    if timestepper == 'runge_kutta':
        transport_scheme = SSPRK3(domain)
        stepper = PrescribedTransport(
            eqn, transport_scheme, io, False, transport_method,
            physics_parametrisations=[instant_rain]
        )
    elif timestepper == 'split':
        transport_scheme = SSPRK3(domain)
        stepper = SplitPrescribedTransport(
            eqn, transport_scheme, io, False, spatial_methods=[transport_method],
            physics_schemes=[(instant_rain, ForwardEuler(domain))],
            physics_frequency=physics_frequency
        )
    else:
        raise ValueError(f"Unknown timestepper option: {timestepper!r}")

    return stepper, x


def run_moisture_experiment(dt, timestepper='split', physics_frequency=1, dirname=None):
    stepper, x = build_stepper(
        dt, timestepper=timestepper, physics_frequency=physics_frequency, dirname=dirname
    )

    stepper.fields('water_vapour').interpolate(vapour_init_expr(x))
    stepper.fields('rain').interpolate(Function(stepper.fields('rain').function_space()))
    Vu = stepper.fields('u').function_space()
    stepper.fields('u').project(as_vector([u0]))

    stepper.run(t=0, tmax=tmax)

    return stepper


def compute_l2_errors(stepper):
    rain_field = stepper.fields('rain')
    vapour_field = stepper.fields('water_vapour')

    x_rain = SpatialCoordinate(rain_field.function_space().mesh())
    x_vap = SpatialCoordinate(vapour_field.function_space().mesh())

    rain_error = errornorm(rain_analytic_expr(x_rain), rain_field)
    vapour_error = errornorm(vapour_analytic_expr(x_vap), vapour_field)

    return rain_error, vapour_error


def plot_rain_comparison(stepper, dirname):
    """
    Plots the final numerical rain field against the analytic rainfall
    profile, saving the figure into the run's results directory.
    """
    rain_field = stepper.fields('rain')
    V = rain_field.function_space()
    x_coords = Function(V).interpolate(SpatialCoordinate(V.mesh())[0]).dat.data_ro
    rain_values = rain_field.dat.data_ro

    order = np.argsort(x_coords)
    x_sorted = x_coords[order]
    rain_sorted = rain_values[order]

    x_fine = np.linspace(0, L, 1000)
    rain_fine = rain_analytic_numpy(x_fine)

    fig, ax = plt.subplots()
    ax.plot(x_fine, rain_fine, '-', color='black', label='Analytic')
    ax.plot(x_sorted, rain_sorted, 'o', color='C0', markersize=3, label='Numerical')
    ax.set_xlabel('x')
    ax.set_ylabel('rain')
    ax.legend()

    os.makedirs(dirname, exist_ok=True)
    fig.savefig(os.path.join(dirname, 'rain_comparison.png'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--timestepper', choices=['runge_kutta', 'split'], default='split')
    parser.add_argument('--physics_frequency', type=int, default=1)
    parser.add_argument('--dirname', type=str, default=None)
    args = parser.parse_args()

    stepper = run_moisture_experiment(
        args.dt, timestepper=args.timestepper,
        physics_frequency=args.physics_frequency, dirname=args.dirname
    )

    rain_error, vapour_error = compute_l2_errors(stepper)

    print(f"dt={args.dt}, timestepper={args.timestepper}, "
          f"physics_frequency={args.physics_frequency}")
    print(f"L2 error (rain):         {rain_error:.6e}")
    print(f"L2 error (water vapour): {vapour_error:.6e}")

    plot_rain_comparison(stepper, stepper.io.dumpdir)
