from firedrake import *
from firedrake.adjoint import *
from gusto import *
continue_annotation()

n = 30
mesh = PeriodicUnitSquareMesh(n, n)
output = OutputParameters(dirname="adjoint_diffusion")
dt = 0.01
domain = Domain(mesh, 10*dt, family="BDM", degree=1)
io = IO(domain, output)

V = VectorFunctionSpace(mesh, "CG", 2)
domain.spaces.add_space("vecCG", V)

R = FunctionSpace(mesh, "R", 0)
# We need to define nu as a function in order to have a control variable.
nu = Function(R, val=0.0001)
diffusion_params = DiffusionParameters(kappa=nu)
eqn = DiffusionEquation(domain, V, "f", diffusion_parameters=diffusion_params)

diffusion_scheme = BackwardEuler(domain)
diffusion_methods = [CGDiffusion(eqn, "f", diffusion_params)]
timestepper = Timestepper(eqn, diffusion_scheme, io, spatial_methods=diffusion_methods)

x = SpatialCoordinate(mesh)
fexpr = as_vector((sin(2*pi*x[0]), cos(2*pi*x[1])))
timestepper.fields("f").interpolate(fexpr)

end = 0.1
timestepper.run(0., end)

u = timestepper.fields("f")
J = assemble(inner(u, u)*dx)

# flag to switch between nu and u for control variable
nu_is_control = True
if nu_is_control:
    my_control = Control(nu)
    h = Function(R, val=0.0001)  # the direction of the perturbation
else:
    my_control = Control(u)
    h = Function(V).interpolate(fexpr)  # the direction of the perturbation

# dJdnu = compute_gradient(J, my_control)

Jhat = ReducedFunctional(J, my_control)  # the functional as a pure function of nu

if nu_is_control:
    # The convergence rate should be around 2.0.
    taylor_test(Jhat, nu, h)
else:
    # The convergence rate should be around 2.0.
    taylor_test(Jhat, u, h)
