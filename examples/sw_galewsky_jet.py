from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, \
    Constant, ge, le, exp, cos
from scipy import pi
import sys

day = 24.*60.*60.
dt = 240.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 6*day

# setup shallow water parameters
R = 6371220.
H = 10000.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

perturb = False
if perturb:
    dirname = "sw_galewsky_jet_perturbed"
else:
    dirname = "sw_galewsky_jet_unperturbed"
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=5, degree=3)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)

timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname=dirname, dumplist_latlon=['D'])

state = State(mesh, horizontal_degree=1,
              family="BDM",
              Coriolis=parameters.Omega,
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist)

# initial conditions
u0 = state.fields("u")
D0 = state.fields("D")

# get lat lon coordinates
theta, lamda = latlon_coords(mesh)

# expressions for meridional and zonal velocity
u_max = Constant(80.)
theta0 = pi/7.
theta1 = pi/2. - theta0
en = exp(-4./((theta1-theta0)**2))
u_zonal_expr = (u_max/en)*exp(1/((theta - theta0)*(theta - theta1)))
u_zonal = conditional(ge(theta, theta0), conditional(le(theta, theta1), u_zonal_expr, 0.), 0.)
u_merid = Constant(0.)

# get cartesian components of velocity
uexpr = sphere_to_cartesian(mesh, u_zonal, u_merid)
u0.project(uexpr)

Rc = Constant(R)
g = Constant(parameters.g)


def D_integrand(th):
    # Initial D field is calculated by integrating D_integrand w.r.t. theta
    from scipy import exp, sin, tan
    f = 2*parameters.Omega*sin(th)
    en = exp(-4./((theta1 - theta0)**2))
    u_zon = (80./en)*exp(1/((th - theta0)*(th - theta1)))
    u_zon[th >= theta1] = 0.
    u_zon[th <= theta0] = 0.
    return -u_zon*(f+tan(th)*u_zon/R)


def Dval(X):
    # Function to return value of D at X
    from scipy import sqrt, arcsin, integrate
    val = []
    for x in X:
        z = R*x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
        th = arcsin(z/R)
        if th <= theta0:
            val.append(0.)
        else:
            v = integrate.quadrature(D_integrand, theta0, th, tol=1.e-8, maxiter=100)
            val.append((Rc/g)*v[0])
    return val

# Get coordinates to pass to Dval function
W = VectorFunctionSpace(mesh, D0.ufl_element())
X = interpolate(mesh.coordinates, W)
D0.dat.data[:] = Dval(X.dat.data_ro)

# Adjust mean value of initial D
C = Function(D0.function_space()).assign(Constant(1.0))
area = assemble(C*dx)
Dmean = assemble(D0*dx)/area
D0 -= Dmean
D0 += Constant(parameters.H)

# optional perturbation
if perturb:
    alpha = Constant(1/3.)
    beta = Constant(1/15.)
    Dhat = Constant(120.)
    theta2 = Constant(pi/4.)
    g = Constant(parameters.g)
    D_pert = Function(D0.function_space()).interpolate(Dhat*cos(theta)*exp(-(lamda/alpha)**2)*exp(-((theta2 - theta)/beta)**2))
    D0 += D_pert

state.initialise({'u': u0, 'D': D0})

ueqn = EulerPoincare(state, u0.function_space())
Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
advection_dict = {}
advection_dict["u"] = ThetaMethod(state, u0, ueqn)
advection_dict["D"] = SSPRK3(state, D0, Deqn)

linear_solver = ShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state)

# build time stepper
stepper = Timestepper(state, advection_dict, linear_solver,
                      sw_forcing)

stepper.run(t=0, tmax=tmax)
