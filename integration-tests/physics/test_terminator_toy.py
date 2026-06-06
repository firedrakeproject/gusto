"""
This tests the terminator toy physics scheme that models the interaction
of two chemical species through coupled ODEs.
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, cos, \
    sin, SpatialCoordinate, Function, max_value, as_vector, \
    errornorm, norm
import numpy as np
import pytest


def run_terminator_toy(dirname, physics_coupling):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # A much larger timestep than in proper simulations, as this
    # tests moving towards a steady state with no flow.
    dt = 50000.

    # Make the mesh and domain
    R = 6371220.
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=2, degree=2)

    # get lat lon coordinates
    x = SpatialCoordinate(mesh)
    lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])

    domain = Domain(mesh, dt, 'BDM', 1)

    # Define the interacting species
    Y = ActiveTracer(name='Y', space='DG',
                     variable_type=TracerVariableType.mixing_ratio,
                     transport_eqn=TransportEquationType.advective)

    Y2 = ActiveTracer(name='Y2', space='DG',
                      variable_type=TracerVariableType.mixing_ratio,
                      transport_eqn=TransportEquationType.advective)

    tracers = [Y, Y2]

    # Equation
    V = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)

    output = OutputParameters(dirname=dirname+"/terminator_toy",
                              dumpfreq=10)
    io = IO(domain, output)

    # Define the reaction rates:
    theta_c = np.pi/9.
    lamda_c = -np.pi/3.

    k1 = max_value(0, sin(theta)*sin(theta_c) + cos(theta)*cos(theta_c)*cos(lamda-lamda_c))
    k2 = 1
    transport_method = [DGUpwind(eqn, 'Y'), DGUpwind(eqn, 'Y2')]
    if physics_coupling == "split":
        physics_schemes = [(TerminatorToy(eqn, k1=k1, k2=k2, species1_name='Y',
                            species2_name='Y2'), BackwardEuler(domain))]

        transport_scheme = SSPRK3(domain)
        time_varying_velocity = True
        stepper = SplitPrescribedTransport(
            eqn, transport_scheme, io, time_varying_velocity,
            spatial_methods=transport_method, physics_schemes=physics_schemes
        )
    elif physics_coupling == "analytic":
        physics_schemes = [(TerminatorToy(eqn, k1=k1, k2=k2, species1_name='Y',
                            species2_name='Y2', analytical_formulation=True),
                            ForwardEuler(domain))]

        transport_scheme = SSPRK3(domain)
        time_varying_velocity = True
        stepper = SplitPrescribedTransport(
            eqn, transport_scheme, io, time_varying_velocity,
            spatial_methods=transport_method, physics_schemes=physics_schemes
        )

    else:
        physics_parametrisation = [TerminatorToy(eqn, k1=k1, k2=k2, species1_name='Y',
                                                 species2_name='Y2')]
        eqn.label_terms(lambda t: not t.has_label(time_derivative), implicit)
        transport_scheme = IMEX_SSP3(domain)
        time_varying_velocity = True
        stepper = PrescribedTransport(
            eqn, transport_scheme, io, time_varying_velocity, transport_method,
            physics_parametrisations=physics_parametrisation
        )

    # Set up a non-divergent, time-varying, velocity field
    def u_t(t):
        return as_vector([Constant(0)*lamda, Constant(0)*lamda, Constant(0)*lamda])

    stepper.setup_prescribed_expr(u_t)

    Y_T_0 = 4e-6
    Y_0 = Y_T_0 + 0*lamda
    Y2_0 = 0*lamda

    stepper.fields("Y").interpolate(Y_0)
    stepper.fields("Y2").interpolate(Y2_0)

    stepper.run(t=0, tmax=10*dt)

    # Compute the steady state solution to compare to
    steady_space = domain.spaces('DG')
    Y_steady = Function(steady_space)
    Y2_steady = Function(steady_space)

    r = k1/(4*k2)
    D_val = sqrt(r**2 + 2*Y_T_0*r)

    Y_steady.interpolate(D_val - r)
    Y2_steady.interpolate(0.5*(Y_T_0 - D_val + r))

    return stepper, Y_steady, Y2_steady


@pytest.mark.parametrize("physics_coupling", ["split", "nonsplit", "analytic"])
def test_terminator_toy_setup(tmpdir, physics_coupling):
    dirname = str(tmpdir)
    stepper, Y_steady, Y2_steady = run_terminator_toy(dirname, physics_coupling)
    Y_field = stepper.fields("Y")
    Y2_field = stepper.fields("Y2")

    print(errornorm(Y_field, Y_steady)/norm(Y_steady))
    print(errornorm(Y2_field, Y2_steady)/norm(Y2_steady))

    # Assert that the physics scheme has sufficiently moved
    # the species fields near their steady state solutions
    assert errornorm(Y_field, Y_steady)/norm(Y_steady) < 0.4, "The Y field is not sufficiently close to the steady state profile"
    assert errornorm(Y2_field, Y2_steady)/norm(Y2_steady) < 0.4, "The Y2 field is not sufficiently close to the steady state profile"
