"""
Test whether the conservative recovery process is working appropriately.
"""

from firedrake import (PeriodicIntervalMesh, IntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, FiniteElement, FunctionSpace,
                       TensorProductElement, Function, interval, norm, errornorm,
                       assemble)
from gusto import *
import numpy as np
import pytest

np.random.seed(0)


@pytest.fixture
def mesh(geometry):

    L = 100.
    H = 100.

    deltax = L / 5.
    deltaz = H / 5.
    nlayers = int(H/deltaz)
    ncolumns = int(L/deltax)

    if geometry == "periodic":
        m = PeriodicIntervalMesh(ncolumns, L)
    elif geometry == "non-periodic":
        m = IntervalMesh(ncolumns, L)

    extruded_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=deltaz)

    return extruded_mesh

def expr(geometry, mesh, configuration):

    x, z = SpatialCoordinate(mesh)

    if configuration == 'rho_constant':
        rho_expr = Constant(2.0)
        if geometry == "periodic":
            m_expr = np.random.randn() + np.random.randn() * z
        elif geometry == "non-periodic":
            m_expr = np.random.randn() + np.random.randn() * x + np.random.randn() * z

    elif configuration == 'm_constant':
        m_expr = Constant(0.01)
        if geometry == "periodic":
            rho_expr = np.random.randn() + np.random.randn() * z
        elif geometry == "non-periodic":
            rho_expr = np.random.randn() + np.random.randn() * x + np.random.randn() * z

    return rho_expr, m_expr

@pytest.mark.parametrize("configuration", ["m_constant", "rho_constant"])
@pytest.mark.parametrize("geometry", ["periodic", "non-periodic"])
def test_conservative_recovery(geometry, mesh, configuration):

    rho_expr, m_expr = expr(geometry, mesh, configuration)

    # construct theta elemnt
    cell = mesh._base_mesh.ufl_cell().cellname()
    w_hori = FiniteElement("DG", cell, 0)
    w_vert = FiniteElement("CG", interval, 1)
    theta_element = TensorProductElement(w_hori, w_vert)

    # spaces
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    DG1 = FunctionSpace(mesh, "DG", 1)
    Vt = FunctionSpace(mesh, theta_element)

    # set up density
    rho_DG1 = Function(DG1).interpolate(rho_expr)
    rho_DG0 = Function(DG0).project(rho_DG1)

    # mixing ratio fields
    m_Vt = Function(Vt).interpolate(m_expr)
    m_DG1_approx = Function(DG1).interpolate(m_expr)
    m_Vt_back = Function(Vt)
    m_DG1 = Function(DG1)

    options = ConservativeRecoveryOptions(embedding_space=DG1,
                                          recovered_space=CG1,
                                          boundary_method=BoundaryMethod.taylor)

    # make the recoverers and do the recovery
    conservative_recoverer = ConservativeRecoverer(m_Vt, m_DG1,
                                                   rho_DG0, rho_DG1, options)
    back_projector = ConservativeProjector(rho_DG1, rho_DG0, m_DG1, m_Vt_back,
                                           subtract_mean=True)

    conservative_recoverer.project()
    back_projector.project()

    # check various aspects of the process
    m_high_diff = errornorm(m_DG1, m_DG1_approx) / norm(m_DG1_approx)
    m_low_diff = errornorm(m_Vt_back, m_Vt) / norm(m_Vt)
    mass_low = assemble(rho_DG0*m_Vt*dx)
    mass_high = assemble(rho_DG1*m_DG1*dx)

    assert (mass_low - mass_high) / mass_high < 5e-14, \
        f'Conservative recovery on {geometry} vertical slice not conservative for {configuration} configuration'
    assert m_low_diff < 2e-14, \
        f'Irreversible conservative recovery on {geometry} vertical slice for {configuration} configuration'

    if configuration in ['m_constant', 'rho_constant']:
        assert m_high_diff < 2e-14, \
            f'Inaccurate conservative recovery on {geometry} vertical slice for {configuration} configuration'
