from gusto import *
from gusto import NumericalIntegral
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi, exp, CubedSphereMesh
import numpy as np

day = 24. * 60. * 60.
ref = 5
dt = 1000
tmax = 5*day
ndumps = 5

# Shallow Water Parameters
a = 6371220.
H = 5960.

parameters = ShallowWaterParameters(H=H)

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Mesh and domain
mesh = CubedSphereMesh(radius=a,
                            refinement_level=ref, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, "BDM", 0)

# Equations
lat, lon = latlon_coords(mesh)
Omega = parameters.Omega
fexpr = 2 * Omega * x[2] / a
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option='vector_advection_form')

# Output and IO
dirname = 'Williamson3_lowest_order'
dumpfreq = int(tmax / (ndumps * dt))
output = OutputParameters(dirname=dirname,
                            dumpfreq=dumpfreq,
                            dumplist_latlon=['D', 'D_error'],
                            log_level='INFO')
diagnostic_fields = [CourantNumber(), SteadyStateError('u'), SteadyStateError('D'), RelativeVorticity()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport Fields and time stepper
# Configuring lowest order fields

VDG1 = domain.spaces("DG1_equispaced")
VCG1 = FunctionSpace(mesh, "CG", 1)
RTCF2 = FunctionSpace(mesh, "RTCF", 2)
RTCE2 = FunctionSpace(mesh, "RTCE", 2)


u_opts = RecoveryOptions(embedding_space=RTCF2,
                         recovered_space=RTCE2)
D_opts = RecoveryOptions(embedding_space=VDG1,
                         recovered_space=VCG1)

transported_fields = [SSPRK3(domain, "u", options=u_opts),
                    SSPRK3(domain, "D", options=D_opts)]

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


def u_func(lat):
    x = xe*(lat - lat_b) / lat_diff
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

# Dbar is a background field for diagnostics
Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])
# ------------------------------------------------------------------------ #
# Run!
# ------------------------------------------------------------------------ #
stepper.run(t=0, tmax=tmax)