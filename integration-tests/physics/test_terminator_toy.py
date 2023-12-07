"""
This tests the terminator toy physics scheme that models the interaction
of two chemical species through coupled ODEs.
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, cos, \
    sin, SpatialCoordinate, Function, max_value, as_vector, \
    errornorm, norm
import numpy as np


def run_terminator_toy(dirname):

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
    X = ActiveTracer(name='X', space='DG',
                     variable_type=TracerVariableType.mixing_ratio,
                     transport_eqn=TransportEquationType.advective)

    X2 = ActiveTracer(name='X2', space='DG',
                      variable_type=TracerVariableType.mixing_ratio,
                      transport_eqn=TransportEquationType.advective)

    tracers = [X, X2]

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

    physics_schemes = [(TerminatorToy(eqn, k1=k1, k2=k2, species1_name='X',
                        species2_name='X2'), BackwardEuler(domain))]

    # Set up a non-divergent, time-varying, velocity field
    def u_t(t):
        return as_vector([Constant(0)*lamda,Constant(0)*lamda,Constant(0)*lamda])

    X_T_0 = 4e-6
    X_0 = X_T_0 + 0*lamda
    X2_0 = 0*lamda

    transport_scheme = SSPRK3(domain)
    transport_method = [DGUpwind(eqn, 'X'), DGUpwind(eqn, 'X2')]

    stepper = SplitPrescribedTransport(eqn, transport_scheme, io,
                                       spatial_methods=transport_method,
                                       physics_schemes=physics_schemes,
                                       prescribed_transporting_velocity=u_t)

    stepper.fields("X").interpolate(X_0)
    stepper.fields("X2").interpolate(X2_0)

    stepper.run(t=0, tmax=10*dt)

    # Compute the steady state solution to compare to
    steady_space = domain.spaces('DG')
    X_steady = Function(steady_space)
    X2_steady = Function(steady_space)

    r = k1/(4*k2)
    D_val = sqrt(r**2 + 2*X_T_0*r)

    X_steady.interpolate(D_val - r)
    X2_steady.interpolate(0.5*(X_T_0 - D_val + r))

    return stepper, X_steady, X2_steady


def test_terminator_toy_setup(tmpdir):
    dirname = str(tmpdir)
    stepper, X_steady, X2_steady = run_terminator_toy(dirname)
    X_field = stepper.fields("X")
    X2_field = stepper.fields("X2")

    print(errornorm(X_field, X_steady)/norm(X_steady))
    print(errornorm(X2_field, X2_steady)/norm(X2_steady))

    # Assert that the physics scheme has sufficiently moved
    # the species fields near their steady state solutions
    assert errornorm(X_field, X_steady)/norm(X_steady) < 0.4, "The X field is not sufficiently close to the steady state profile"
    assert errornorm(X2_field, X2_steady)/norm(X2_steady) < 0.4, "The X2 field is not sufficiently close to the steady state profile"
