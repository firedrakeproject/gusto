"""
An implementation of the Williams 3 Test case with
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi, exp, errornorm
import numpy as np
import pickle


# Set up timestepping variables
day = 24. * 60. * 60.
ref = [3, 4, 5, 6]
dt = [4000, 2000, 1000, 500]
tmax = 5*day
ndumps = 5

# Shallow Water Parameters
a = 6371220.
H = 5960.

error = []
ref_level = []
dt_step = []

parameters = ShallowWaterParameters(H=H)


# Starts for loop for different reginement levels
for i in range(len(ref)):
    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Mesh and domain
    mesh = IcosahedralSphereMesh(radius=a,
                                 refinement_level=ref[i], degree=1)
    x = SpatialCoordinate(mesh)
    global_normal = x
    mesh.init_cell_orientations(x)
    domain = Domain(mesh, dt[i], "BDM", 1)

    # Equations
    lat, lon = latlon_coords(mesh)
    Omega = parameters.Omega
    fexpr = 2*Omega * x[2] / a
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option='vector_advection_form')

    # Output and IO
    dirname = 'convergenceplotting_ref_%s' % ref[i]
    dumpfreq = int(tmax / (ndumps*dt[i]))
    output = OutputParameters(dirname=dirname,
                              dumpfreq=dumpfreq,
                              dumplist_latlon=['D', 'D_error'],
                              log_level='INFO')
    diagnostic_fields = [CourantNumber(), SteadyStateError('u'), SteadyStateError('D'), RelativeVorticity()]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport Fields and time stepper
    transported_fields = [SSPRK3(domain, "u"),
                          SSPRK3(domain, "D")]

    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields)

    # ------------------------------------------------------------------------ #
    # Initial Conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')

    # Get Vector co-ordinate for the u expression
    e_x = as_vector([Constant(1.0), Constant(0.0), Constant(0.0)])
    e_y = as_vector([Constant(0.0), Constant(1.0), Constant(0.0)])
    R = sqrt(x[0]**2 + x[1]**2)

    e_lon = (x[0] * e_y - x[1]*e_x) / R

    xe = 0.3
    u_0 = 2 * pi * a / (12*day)
    lat_b = -pi / 6
    lat_e = pi / 2
    g = parameters.g
    h0 = 2.94e4 / g
    x_mod = xe*(lat - lat_b)/(lat_e - lat_b)
    en = exp(4/xe)
    lat_diff = lat_e - lat_b

    # -------------------------------------------------------------------- #
    # Obtain u and D (by integration of analytic expression)
    # -------------------------------------------------------------------- #

    uexpr = conditional(x_mod <= 0, 0.0,
                        conditional(x_mod >= xe, 0.0,
                                    (u_0*en * exp(-1/x_mod) * exp(-1/(xe-x_mod)))
                                    )
                        )

    def u_func(y):
        x = xe*(y - lat_b) / lat_diff
        very_small = 1e-9
        return np.where(x <= 0, very_small,
                        np.where(x >= xe, very_small,
                                 u_0 * en * np.exp(xe / (x * (x - xe)))
                                 )
                        )

    def h_func(y):
        return a/g*(2*Omega*np.sin(y) + u_func(y)*np.tan(y)/a)*u_func(y)

    lat_VD = Function(D0.function_space()).interpolate(lat)
    D0_integral = Function(D0.function_space())
    h_integral = NumericalIntegral(-pi/2, pi/2)
    h_integral.tabulate(h_func)
    D0_integral.dat.data[:] = h_integral.evaluate_at(lat_VD.dat.data[:])
    Dexpr = h0 - D0_integral

    u0.project(as_vector(e_lon * uexpr))
    D0.interpolate(Dexpr)

    Vu = stepper.fields("u")
    uexact = Function(Vu)
    uexact.interpolate(u0)
    # Dbar is a background field for diagnostics
    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])
    # ------------------------------------------------------------------------ #
    # Run!
    # ------------------------------------------------------------------------ #
    stepper.run(t=0, tmax=tmax)

    u = stepper.fields("u")
    L2_error = errornorm(uexact, u)
    error.append(L2_error)
    dt_step.append(dt[i])
    ref_level.append(ref[i])
# Makes use of Pickle to save data
plotdata = [error, ref_level, dt_step]
with open('plotdata', 'wb') as f:
    pickle.dump(plotdata, f)
