"""
# This tests the SW_AdjustableSaturation physics class. In the first scenario
# it creates a cloud in a subsaturated atmosphere that should evaporate.
# In the second test it creates a bubble of water vapour that is advected by
# a prescribed velocity and should be converted to cloud where it exceeds a
# saturation threshold. The first test passes if the cloud is zero, the
# vapour has increased, the buoyancy field has increased and the total moisture
# is conserved. The second test passes if cloud is non-zero, vapour has
# decreased, buoyancy has decreased and total moisture is conserved.
"""

from os import path
from gusto import *
from firedrake import (IcosahedralSphereMesh, acos, sin, cos, Constant, norm)
import pytest


def run_sw_cond_evap(dirname, process):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Parameters
    dt = 100
    R = 6371220.
    H = 100
    theta_c = pi
    lamda_c = pi/2
    rc = R/4
    L = 10
    beta2 = 1

    # Domain
    mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=2)
    degree = 1
    domain = Domain(mesh, dt, 'BDM', degree)
    x = SpatialCoordinate(mesh)
    theta, lamda = latlon_coords(mesh)

    # saturation field (constant everywhere)
    sat = 100

    # Equation
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R

    tracers = [WaterVapour(space='DG'), CloudWater(space='DG')]

    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                 u_transport_option='vector_advection_form',
                                 thermal=True, active_tracers=tracers)

