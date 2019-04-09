from gusto import *
from firedrake import PeriodicRectangleMesh, SpatialCoordinate, \
    Function, grad, as_vector, sin, pi, FunctionSpace, Constant, \
    CellVolume, inner, exp

# set up PUSM parameters and mesh
res = 256
dt = 1/24./60./2.
tmax =  10
maxk = 6
dumpfreq = 30
vorticity = True
vorticity_SUPG = True

# set up fields
fieldlist = ['u', 'D']
if vorticity:
    fieldlist.append('q')

g = 9.81*(24.*3600.)**2./1000. # in km/days^2
theta = 25./360.*2.*pi # latitude
f = 2.*sin(theta)*2.*pi  # in 1/days
H = 10.
parameters = ShallowWaterParameters(g=g, H=H)

dirname = ("EC_div_vort_vort{0}_SUPG{1}_res{2}"\
           "_dt{3}_maxk{4}".format(int(vorticity), int(vorticity_SUPG),
                                          res, round(dt, 4), maxk))
Lx, Ly = 5000, 4330.
mesh = PeriodicRectangleMesh(res, res, Lx, Ly)

timestepping = TimesteppingParameters(dt=dt, alpha=1., maxk=maxk)
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
psiexpr =  - DeltaH*(e1 + e2 - 4.*pi*sx*sy/Lx/Ly)*g/f

# Psi should live in vorticity space
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
    if vorticity_SUPG == True:
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
    upwind = True

    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("D", ThetaMethod(state, D0, Deqn)))

linear_solver = ShallowWaterSolver(state)

# Set up forcing
sw_forcing = HamiltonianShallowWaterForcing(state, upwind=upwind,
                                            euler_poincare=False)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver, sw_forcing)

stepper.run(t=0, tmax=tmax)
