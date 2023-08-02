"""
Tests the CoupledTransportEquation() class.
Two tracers -- a mixing ratio and a density -- are transported using
combinations of advective and conservative forms.


"""

from firedrake import norm, VectorFunctionSpace, as_vector, max_value
from gusto import *
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    norm1 = norm(timestepper.fields("f1") - f_end) / norm(f_end)
    norm2 = norm(timestepper.fields("f2") - f_end) / norm(f_end)
    return (norm1 + norm2)/2


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form1", ["advective", "continuity"])
@pytest.mark.parametrize("equation_form2", ["advective", "continuity"])
def test_coupled_transport_scalar(tmpdir, geometry, equation_form1, equation_form2, tracer_setup):
    setup = tracer_setup(tmpdir, geometry)
    domain = setup.domain
    
    if equation_form1 == "advective":
      tracer1 = ActiveTracer(name = 'f1', space = 'DG', 
                            variable_type = TracerVariableType.density,
                            transport_eqn = TransportEquationType.advective)                  
    else:
      tracer1 = ActiveTracer(name = 'f1', space = 'DG', 
                            variable_type = TracerVariableType.density,
                            transport_eqn = TransportEquationType.conservative) 
                            
    if equation_form2 == "advective":
      tracer2 = ActiveTracer(name = 'f2', space = 'DG', 
                            variable_type = TracerVariableType.mixing_ratio,
                            transport_eqn = TransportEquationType.advective)                   
    else:
      tracer2 = ActiveTracer(name = 'f2', space = 'DG', 
                            variable_type = TracerVariableType.mixing_ratio,
                            transport_eqn = TransportEquationType.conservative) 
                            
    tracers = [tracer1,tracer2]

    V = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu = V)
    
    transport_scheme = SSPRK3(domain)
    #transport_method = DGUpwind(eqn, for tracer in tracers)
    transport_method = [DGUpwind(eqn, 'f1'), DGUpwind(eqn, 'f2')]

    timestepper = PrescribedTransport(eqn, transport_scheme, setup.io, transport_method)

    # Initial conditions
    timestepper.fields("f1").interpolate(setup.f_init)
    timestepper.fields("f2").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)

    error = run(timestepper, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'