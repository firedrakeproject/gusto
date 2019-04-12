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
    square_test = False
vorticity = True
vorticity_SUPG = False

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

nq = '' if vorticity else 'no'
nqSUPG = '' if vorticity_SUPG else 'no'
dirname = ("EC_{0}_{1}q_{2}qSUPG_res{3}_dt{4}_maxk"
           "{5}".format(fname, nq, nqSUPG, res, round(dt, 4), maxk))

timestepping = TimesteppingParameters(dt=dt, maxk=maxk)
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
              hamiltonian=True,
              reconstruct_q=True,
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

    # Psi should live in vorticity/stream function space
    if 'q' in fieldlist:
        q0 = state.fields('q')
        Vq = q0.function_space()
    else:
        Vq = FunctionSpace(mesh, "CG", 3)
    psi = Function(Vq).interpolate(psiexpr)
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

if vorticity:
    # initial q solver
    initial_vorticity(state, D0, u0, q0, f)

    Deqn = AdvectionEquation(state, D0.function_space(),
                             ibp=IntegrateByParts.NEVER,
                             equation_form="continuity", flux_form=True)
    upwind_D = False

    if vorticity_SUPG:
        # set up vorticity SUPG parameter
        cons, vol, eps = Constant(0.1), CellVolume(mesh), 1.0e-10
        Fmag = (inner(D0*u0, D0*u0) + eps)**0.5
        q_SUPG = SUPGOptions()
        q_SUPG.default = (cons/Fmag)*vol**0.5
        q_SUPG.constant_tau = False

        qeqn = SUPGAdvection(state, q0.function_space(),
                             ibp=IntegrateByParts.NEVER,
                             supg_params=q_SUPG, flux_form=True)
    else:
        qeqn = AdvectionEquation(state, q0.function_space(),
                                 ibp=IntegrateByParts.NEVER, flux_form=True)

    advected_fields = []
    # flux formulation has Dp in q-eqn, qp in u-eqn, so order matters
    advected_fields.append(("D", ForwardEuler(state, D0, Deqn)))
    advected_fields.append(("q", ThetaMethod(state, q0, qeqn, weight='D')))

    # Advected fields building q equation sets up q SUPG, which is needed
    # in ueqn.
    ueqn = VectorInvariant(state, u0.function_space(), vorticity=True)
    advected_fields.append(("u", ForwardEuler(state, u0, ueqn)))
    upwind = False
else:
    ueqn = VectorInvariant(state, u0.function_space())
    Deqn = AdvectionEquation(state, D0.function_space(),
                             equation_form="continuity")
    upwind_D = True

    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("D", ThetaMethod(state, D0, Deqn)))

linear_solver = ShallowWaterSolver(state)

# Set up forcing
sw_forcing = HamiltonianShallowWaterForcing(state, upwind_d=upwind_D,
                                            euler_poincare=False)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver, sw_forcing)

stepper.run(t=0, tmax=tmax)
