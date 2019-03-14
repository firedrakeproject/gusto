from os import path
from gusto import *
from firedrake import (PeriodicUnitSquareMesh, pi, PeriodicIntervalMesh,
                       ExtrudedMesh, as_vector, sin, SpatialCoordinate,
                       Constant, FunctionSpace, Function, sqrt, conditional, cos)
from netCDF4 import Dataset
import pytest

def setup_SWE_test(dirname):

    res = 10
    dt = 0.001
    tmax = 0.05
    maxk = 8
    f, g, H = 5., 5., 1.

    mesh = PeriodicUnitSquareMesh(res, res)

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters(dt=dt, alpha=1., maxk=maxk)
    output = OutputParameters(dirname=dirname+"/energy_SWE", dumpfreq=1)
    parameters = ShallowWaterParameters(g=g, H=H)
    diagnostic_fields = [ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(),
                         Sum("ShallowWaterKineticEnergy",
                             "ShallowWaterPotentialEnergy")]

    state = State(mesh, horizontal_degree=1,
                  family="BDM",
                  hamiltonian=True,
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = state.fields('u')
    D0 = state.fields('D')
    x = SpatialCoordinate(mesh)

    uexpr = as_vector([0.0, sin(2*pi*x[0])])
    Dexpr = H + 1/(4*pi)*f/g*sin(4*pi*x[1])

    # Coriolis
    fexpr = Constant(f)
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", V)
    f.interpolate(fexpr)  # Coriolis frequency (1/s)

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    state.initialise([('u', u0),
                      ('D', D0)])

    ueqn = VectorInvariant(state, u0.function_space())
    Deqn = AdvectionEquation(state, D0.function_space(),
                             equation_form="continuity")
    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("D", ThetaMethod(state, D0, Deqn)))
    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = HamiltonianShallowWaterForcing(state, euler_poincare=False)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver, sw_forcing)

    return stepper, tmax


def setup_Euler_test(dirname):
    res = [10, 10]
    dt = 5.
    tmax = 200
    maxk = 8
    gauss_deg = 6

    fieldlist = ['u', 'rho', 'theta']

    H, L = 6400., 12800.
    parameters = CompressibleParameters()
    diagnostics = Diagnostics("CompressibleEnergy")

    nlayers, columns = res[0], res[1]
    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    timestepping = TimesteppingParameters(dt=dt, alpha=1., maxk=maxk)
    output = OutputParameters(dirname=dirname+"/energy_Euler", dumpfreq=1,
                              dumplist=['u'], perturbation_fields=['theta', 'rho'])
    diagnostic_fields = [CompressibleEnergy()]

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  hamiltonian=True,
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = state.fields('u')
    rho0 = state.fields('rho')
    theta0 = state.fields('theta')

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Isentropic background state
    Tsurf = Constant(300.)

    theta_b = Function(Vt).interpolate(Tsurf)
    rho_b = Function(Vr)

    # Calculate hydrostatic Pi
    compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)

    x = SpatialCoordinate(mesh)
    xc = 0.5*L
    xr = 4000.
    zc = 3000.
    zr = 2000.
    r = sqrt(((x[0]-xc)/xr)**2 + ((x[1]-zc)/zr)**2)
    theta_pert = conditional(r > 1., 0., -7.5*(1.+cos(pi*r)))
    theta0.interpolate(theta_b + theta_pert)
    compressible_hydrostatic_balance(state, theta0, rho0, solve_for_rho=True)

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0)])
    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = SUPGAdvection(state, Vt, equation_form="advective")
    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("rho", ThetaMethod(state, rho0, rhoeqn)))
    advected_fields.append(("theta", ThetaMethod(state, theta0, thetaeqn)))

    linear_solver = HybridizedCompressibleSolver(state)

    # Set up forcing
    compressible_forcing = HamiltonianCompressibleForcing(state, gauss_deg=gauss_deg)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields,
                            linear_solver, compressible_forcing)

    return stepper, tmax


def run_energy_test(dirname, test_case):

    if test_case == "SWE":
        stepper, tmax = setup_SWE_test(dirname)
    elif test_case == "Euler":
        stepper, tmax = setup_Euler_test(dirname)    
    stepper.run(t=0, tmax=tmax)


def test_energy_conservation(tmpdir):
    dirname = str(tmpdir)
    run_energy_test(dirname, "SWE")
    filename = path.join(dirname, "energy_SWE/diagnostics.nc")
    data = Dataset(filename, "r")
    En_err_SWE = data.groups["ShallowWaterKineticEnergy_plus_ShallowWaterPotentialEnergy"]

    run_energy_test(dirname, "Euler")
    filename = path.join(dirname, "energy_Euler/diagnostics.nc")
    data = Dataset(filename, "r")
    En_err_Euler = data.groups["CompressibleEnergy"]

    assert max([abs((En_err_SWE["total"][i] - En_err_SWE["total"][0])/En_err_SWE["total"][i]) for i in range(len(En_err_SWE["total"]))]) < 5.e-13
    assert max([abs((En_err_Euler["total"][i] - En_err_Euler["total"][0])/En_err_Euler["total"][i]) for i in range(len(En_err_Euler["total"]))]) < 1.e-7

