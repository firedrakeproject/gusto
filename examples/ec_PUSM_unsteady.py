from gusto import *
from firedrake import PeriodicUnitSquareMesh, SpatialCoordinate, \
    as_vector, sin, pi, FunctionSpace, Constant, CellVolume, inner

# set up PUSM parameters and mesh
res = 24
dt = 0.001
tmax = 1.
maxk = 20
dumpfreq = 10
vorticity = True
vorticity_SUPG = False

# set up fields
fieldlist = ['u', 'D']
if vorticity:
    fieldlist.append('q')

f, g, H = 5., 5., 1.
parameters = ShallowWaterParameters(g=g, H=H)

dirname = ("EC_PUSM_vort{0}_SUPG{1}_res{2}"\
           "_dt{3}_maxk{4}".format(int(vorticity), int(vorticity_SUPG),
                                   res, dt, maxk))
mesh = PeriodicUnitSquareMesh(res, res)

timestepping = TimesteppingParameters(dt=dt, alpha=1., maxk=maxk)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
diagnostics = Diagnostics('D')
diagnostic_fields = [ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(),
                     Sum("ShallowWaterKineticEnergy",
                         "ShallowWaterPotentialEnergy"),
                     PotentialVorticity(),
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

uexpr = as_vector([0.0, sin(2*pi*x[0])])
Dexpr = H + 1/(4*pi)*f/g*sin(4*pi*x[1])

u0.project(uexpr)
D0.interpolate(Dexpr)

state.initialise([('u', u0),
                  ('D', D0)])

# Coriolis
fexpr = Constant(f)
V = FunctionSpace(mesh, "CG", 1)
f = state.fields("coriolis", V)
f.interpolate(fexpr)

if vorticity:
    # initial q solver
    q0 = state.fields('q')
    initial_vorticity(state, D0, u0, q0, f)

    Deqn = AdvectionEquation(state, D0.function_space(),
                             ibp=IntegrateByParts.NEVER,
                             equation_form="continuity", flux_form=True)
    if vorticity_SUPG == True:
        # set up vorticity SUPG parameter
        cons, vol, eps = Constant(0.1), CellVolume(mesh), 1.0e-10
        Fmag = (inner(D0*u0,D0*u0) + eps)**0.5
        q_SUPG = SUPGOptions()
        q_SUPG.default = (cons/Fmag)*vol**0.5
        q_SUPG.constant_tau = False

        qeqn = SUPGAdvection(state, q0.function_space(),
                             ibp=IntegrateByParts.NEVER,
                             supg_params=q_SUPG, flux_form=True)
    else:
        qeqn = AdvectionEquation(state, q0.function_space(),
                                 ibp=IntegrateByParts.NEVER, flux_form=True)
    ueqn = VectorInvariant(state, u0.function_space(), vorticity=True)
    upwind = False

    advected_fields = []
    # flux formulation has Dp in q-eqn, qp in u-eqn, so order matters
    advected_fields.append(("D", ForwardEuler(state, D0, Deqn)))
    advected_fields.append(("q", ThetaMethod(state, q0, qeqn, weight='D')))
    advected_fields.append(("u", ForwardEuler(state, u0, ueqn)))
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
