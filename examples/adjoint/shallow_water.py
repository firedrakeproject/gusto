from firedrake import *
from firedrake.adjoint import *
from gusto import *
continue_annotation()

# setup shallow water parameters
R = 6371220.
H = 5960.
dt = 900.

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)
parameters = ShallowWaterParameters(H=H)

# Equation
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
lsq = (lamda - lamda_c)**2
theta_c = pi/6.
thsq = (theta - theta_c)**2
rsq = min_value(R0sq, lsq+thsq)
r = sqrt(rsq)
bexpr = 2000 * (1 - r/R0)
eqn = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr)

# I/O
output = OutputParameters(dirname="adjoint_sw", log_courant=False)
io = IO(domain, output)

# Transport schemes
transported_fields = [TrapeziumRule(domain, "u"), SSPRK3(domain, "D")]
transport_methods = [DGUpwind(eqn, "u"), DGUpwind(eqn, "D")]

# Time stepper
stepper = SemiImplicitQuasiNewton(
    eqn, io, transported_fields, transport_methods
)

u0 = stepper.fields('u')
D0 = stepper.fields('D')
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

u0.project(uexpr)
D0.interpolate(Dexpr)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

stepper.run(0., 10*dt)

J = assemble(0.5*inner(u0, u0)*dx + 0.5*g*D0**2*dx)

Jhat = ReducedFunctional(J, Control(D0))

conv_rate = taylor_test(Jhat, D0, Function(D0.function_space()).assign(Dexpr))
