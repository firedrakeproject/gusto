from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #
day = 24 * 60 * 60
# setup resolution and timestepping parameters for convergence test
ref = 5
timestep = [50, 100, 150, 200, 225, 250, 275, 300]
tmax = 1 * day
ndumps = 1
steppers = [SSPRK3, RK4, Heun]

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(H=H)

for scheme in steppers:
    for dt in timestep:
        # ------------------------------------------------------------------------ #
        # Set up model objects
        # ------------------------------------------------------------------------ #

        # Domain
        mesh = IcosahedralSphereMesh(radius=R,
                                    refinement_level=ref, degree=2)
        x = SpatialCoordinate(mesh)
        global_normal = x
        mesh.init_cell_orientations(x)
        domain = Domain(mesh, dt, 'BDM', 1)

        # Equation
        Omega = parameters.Omega
        fexpr = 2 * Omega * x[2] / R
        eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option='vector_manifold_advection_form')

        # I/O
        dirname = f'Williams2_{scheme.__name__}_dt={dt}_degree=2'
        dumpfreq = 1 # int(tmax / (ndumps*dt))
        output = OutputParameters(dirname=dirname,
                                dumpfreq=dumpfreq,
                                dumplist_latlon=['D', 'D_error'],
                                log_level='INFO')

        diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                            SteadyStateError('u'), SteadyStateError('D')]
        io = IO(domain, output, diagnostic_fields=diagnostic_fields)

        # Time stepper
        stepper = Timestepper(eqns, scheme(domain), io)

        # ------------------------------------------------------------------------ #
        # Initial conditions
        # ------------------------------------------------------------------------ #

        u0 = stepper.fields("u")
        D0 = stepper.fields("D")
        x = SpatialCoordinate(mesh)
        u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
        uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
        g = parameters.g
        Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

        u0.project(uexpr)
        D0.interpolate(Dexpr)

        Dbar = Function(D0.function_space()).assign(H)
        stepper.set_reference_profiles([('D', Dbar)])

        # ------------------------------------------------------------------------ #
        # Run
        # ------------------------------------------------------------------------ #

        stepper.run(t=0, tmax=tmax)