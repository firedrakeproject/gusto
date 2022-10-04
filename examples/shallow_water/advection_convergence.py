from gusto import *
from firedrake import (PeriodicIntervalMesh, SpatialCoordinate, FunctionSpace,
                       VectorFunctionSpace, conditional, acos, cos, pi, plot,
                       FiniteElement, as_vector, errornorm)
import matplotlib.pyplot as plt

tophat = False
triangle = False
trig = True

# set up resolution and timestepping parameters for convergence test
dx_dt = {0.05: 0.005, 0.1: 0.01, 0.2: 0.02, 0.25: 0.025, 0.5, 0.05}
error_norms = []
dx_list = []
dt_list = []

# set up input that doesn't change with dx or dt
u_max = 1
if tophat:
    qmax = 0.7
    qh = 0.2
elif triangle:
    qmax = 0.9
    qh = 0.4
elif trig:
    C0 = 0.6
    K0 = 0.3
Csat = 0.75
Ksat = 0.25

# loop over range of dx, dt pairs
for dx, dt in dx_dt.items():

    if tophat:
        dirname = "convergence_test_advection_hat_DG1_dx%s_dt%s" % (dx, dt)
    elif triangle:
        dirname = "convergence_test_advection_triangle_DG1_dx%s_dt%s" % (dx, dt)
    elif trig:
        dirname = "compare_exact_convergence_test_advection_trig_DG1_dx%s_dt%s" % (dx, dt)

    Lx = 100
    nx = int(Lx/dx)
    mesh = PeriodicIntervalMesh(nx, Lx)
    x = SpatialCoordinate(mesh)[0]
    x1 = 0
    x2 = Lx/4
    tmax = 55

    output = OutputParameters(dirname=dirname, dumpfreq=100)

    diagnostic_fields = [CourantNumber()]

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=None,
                  diagnostics=None,
                  diagnostic_fields=diagnostic_fields)

    # set up function spaces
    eltDG = FiniteElement("DG", "interval", 1, variant="equispaced")
    VD = FunctionSpace(mesh, eltDG)
    Vu = VectorFunctionSpace(mesh, "CG", 1)

    # initial advected profile
    if tophat:
        mexpr = conditional(x < x2, conditional(x > x1, qmax, qmax-qh), qmax-qh)
        exact_mexpr = conditional(x < (x2+tmax), conditional(x > (x1+tmax), qmax, qmax-qh), qmax-qh)
    elif triangle:
        mexpr = conditional(x < x2, conditional(x > x1, qmax - 2*qh - (4*qh*(x-(Lx/2)))/Lx, qmax-qh), qmax-qh)
    elif trig:
        mexpr = C0 + K0*cos((2*pi*x)/Lx)
        exact_mexpr = C0 + K0*cos((2*pi*(x-(u_max*tmax)))/Lx)

    # set up advection equation
    meqn = AdvectionEquation(state, VD, field_name="advected_m", Vu=Vu)
    state.fields("u").project(as_vector([u_max]))
    state.fields("advected_m").project(mexpr)
    # analytical solution
    exact_m = state.fields("exact_m", VD)
    exact_m.interpolate(exact_mexpr)

    # build time stepper
    stepper = PrescribedTransport(state,
                                  ((meqn,
                                    SSPRK3(state,)),),)
    stepper.run(t=0, tmax=tmax)


    
    fig, axes = plt.subplots()
    plot(exact_m, axes=axes, label='exact solution', color='green')
    plot(state.fields("advected_m"), axes=axes, label='m after advection', color='red')
    plt.title("Profile after advection")
    plt.show()

    # calculate L2 error norm
    m = state.fields("advected_m")
    L2_error = errornorm(exact_m, m)
    print(L2_error)
    error_norms.append(L2_error)
    dx_list.append(dx)
    dt_list.append(dt)

plt.plot(dt_list, error_norms)
plt.xlabel("dt")
plt.title("Errors against dt for constant Courant number")
plt.show()

plt.plot(dx_list, error_norms)
plt.xlabel("dx")
plt.title("Errors against dx for constant Courant number")
plt.show()



    
