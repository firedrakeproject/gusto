from gusto import *
from firedrake import PeriodicRectangleMesh, PeriodicUnitSquareMesh, \
    SpatialCoordinate, Function, grad, as_vector, sin, pi, FunctionSpace, \
    Constant, exp
import sys

dt = 1/24./60./2.
if '--running-tests' in sys.argv:
    tmax = 5*dt
    res = 32
    square_test = False
else:
    tmax = 10.
    res = 256
    square_test = True
hamiltonian = True
upwind_D = False
vorticity = True

# set up parameters and mesh
if square_test:
    res = 24
    dt = 0.001
    tmax = 0.1
    maxk = 20
    dumpfreq = 10
    fname = 'PUSM'

    f, g, H = 5., 5., 1.
    mesh = PeriodicUnitSquareMesh(res, res)
else:
    maxk = 6
    dumpfreq = 30
    fname = 'div_vort'

    g = 9.81*(24.*3600.)**2./1000.  # in km/days^2
    theta = 25./360.*2.*pi  # latitude
    f = 2.*sin(theta)*2.*pi  # in 1/days
    H = 10.
    Lx, Ly = 5000, 4330.
    mesh = PeriodicRectangleMesh(res, res, Lx, Ly)

# set up fields
fieldlist = ['u', 'D']
if vorticity:
    fieldlist.append('q')
# Set up objects for state
parameters = ShallowWaterParameters(g=g, H=H)

upw = '' if upwind_D else 'no'
vort = '_vorticity' if vorticity else ''
ham = '_hamiltonian' if hamiltonian else ''
dirname = ("EC_{0}{1}{2}_{3}upwindD_res{4}_dt{5}_maxk"
           "{6}".format(fname, ham, vort, upw, res, round(dt, 4), maxk))
timestepping = TimesteppingParameters(dt=dt, maxk=maxk)
if hamiltonian:
    hamiltonian = HamiltonianOptions(no_u_rec=vorticity)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq,
                          log_level='INFO')
diagnostics = Diagnostics('D')
diagnostic_fields = [ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(),
                     Sum("ShallowWaterKineticEnergy",
                         "ShallowWaterPotentialEnergy"),
                     PotentialVorticity(),
                     AbsoluteVorticity(),
                     ShallowWaterPotentialEnstrophy()]

state = State(mesh, horizontal_degree=1,
              family="BDM",
              hamiltonian=hamiltonian,
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              diagnostic_fields=diagnostic_fields,
              fieldlist=fieldlist)

# interpolate initial conditions
u0 = state.fields('u')
D0 = state.fields('D')

x = SpatialCoordinate(mesh)
if square_test:
    uexpr = as_vector([0.0, sin(2*pi*x[0])])
    Dexpr = H + 1/(4*pi)*f/g*sin(4*pi*x[1])
    D0.interpolate(Dexpr)
    u0.project(uexpr)
else:
    DeltaH, o = 0.075, 0.1

    xc1, yc1 = (1./2. - o)*Lx, (1./2. - o)*Ly
    xc2, yc2 = (1./2. + o)*Lx, (1./2. + o)*Ly
    sx, sy = 1.5*Lx/20., 1.5*Ly/20.

    xp1 = Lx*sin(pi*(x[0] - xc1)/Lx)/sx/pi
    yp1 = Ly*sin(pi*(x[1] - yc1)/Ly)/sy/pi
    xp2 = Lx*sin(pi*(x[0] - xc2)/Lx)/sx/pi
    yp2 = Ly*sin(pi*(x[1] - yc2)/Ly)/sy/pi

    e1 = exp(-1./2.*(xp1**2. + yp1**2.))
    e2 = exp(-1./2.*(xp2**2. + yp2**2.))
    psiexpr = - DeltaH*(e1 + e2 - 4.*pi*sx*sy/Lx/Ly)*g/f

    # Psi should live in stream function space
    Vpsi = FunctionSpace(mesh, "CG", 3)
    psi = Function(Vpsi).interpolate(psiexpr)
    D0.project(f*psi/g)
    D0 += H
    psi_grad = Function(u0.function_space()).project(grad(psi))
    u0.project(as_vector([-psi_grad[1], psi_grad[0]]))

state.initialise([('u', u0),
                  ('D', D0)])

# Coriolis
fexpr = Constant(f)
V = FunctionSpace(mesh, "CG", 1)
f = state.fields("coriolis", V)
f.interpolate(fexpr)

advected_fields = []
if upwind_D:
    Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
    advected_fields.append(("D", ThetaMethod(state, D0, Deqn)))
else:
    Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity",
                             flux_form=True)
    advected_fields.append(("D", ForwardEuler(state, D0, Deqn)))

# Euler Poincare split only if advection, forcing weights are equal
if upwind_D != vorticity:
    euler_poincare = True
    U_transport = EulerPoincare
else:
    euler_poincare = False
    U_transport = VectorInvariant

if vorticity:
    # initial q solver
    q0 = state.fields('q')
    initial_vorticity(state, D0, u0, q0)

    # flux formulation has Dp in q-eqn, qp in u-eqn, so order matters
    qeqn = AdvectionEquation(state, q0.function_space(),
                             ibp=IntegrateByParts.NEVER, flux_form=True)
    advected_fields.append(("q", ThetaMethod(state, q0, qeqn, weight='D')))

    ueqn = U_transport(state, u0.function_space(), vorticity=True)
    advected_fields.append(("u", ForwardEuler(state, u0, ueqn)))
else:
    ueqn = U_transport(state, u0.function_space())
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))

linear_solver = ShallowWaterSolver(state)

# Set up forcing
if hamiltonian:
    sw_forcing = HamiltonianShallowWaterForcing(state, upwind_d=upwind_D,
                                                euler_poincare=euler_poincare,
                                                vorticity=vorticity)
else:
    sw_forcing = ShallowWaterForcing(state, euler_poincare=euler_poincare)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver, sw_forcing)

stepper.run(t=0, tmax=tmax)
