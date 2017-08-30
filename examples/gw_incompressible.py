from gusto import *
from firedrake import as_vector,\
    sin, SpatialCoordinate, Function
import numpy as np
import sys

dt = 6.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 3600.

ncolumns = 300  # number of columns
nlayers = 10  # horizontal layers
L = 3.0e5
H = 1.0e4  # Height position of the model top
physical_domain = VerticalSlice(H=H, L=L, ncolumns=ncolumns, nlayers=nlayers)

# class containing timestepping parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
timestepping = TimesteppingParameters(dt=dt)

# class containing output parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
output = OutputParameters(dirname='gw_incompressible', dumpfreq=10, dumplist=['u'], perturbation_fields=['b'])

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber()]

# setup state, passing in the mesh
state = IncompressibleEulerState(physical_domain.mesh,
                                 output=output,
                                 diagnostic_fields=diagnostic_fields)

# Set up advection schemes
advected_fields = []
supg = True
if not supg:
    beqn = EmbeddedDGAdvection(state, Vb,
                               equation_form="advective")
    advected_fields.append((("b", SSPRK3(state.fields("b"),
                                     timestepping.dt, beqn))))

model = IncompressibleEulerModel(state, physical_domain, is_rotating=False,
                                 timestepping=timestepping,
                                 advected_fields=advected_fields)

# Initial conditions
# set up functions on the spaces constructed by state
u0 = state.fields("u")
b0 = state.fields("b")
p0 = state.fields("p")

# spaces
Vu = u0.function_space()
Vb = b0.function_space()

x, z = SpatialCoordinate(physical_domain.mesh)

# first setup the background buoyancy profile
# z.grad(bref) = N**2
# the following is symbolic algebra, using the default buoyancy frequency
# from the parameters class.
N = model.parameters.N
bref = z*(N**2)
# interpolate the expression to the function
b_b = Function(Vb).interpolate(bref)

# setup constants
a = 5.0e3
deltab = 1.0e-2
b_pert = deltab*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
# interpolate the expression to the function
b0.interpolate(b_b + b_pert)

incompressible_hydrostatic_balance(state, physical_domain.vertical_normal, b_b, p0)

u0.project(as_vector([20.0, 0.0]))

# pass these initial conditions to the state.initialise method
state.initialise([('u', u0),
                  ('b', b0)])

# set the background buoyancy
state.set_reference_profiles([('b', b_b)])

# build time stepper
stepper = Timestepper(model)

# run timestepper
stepper.run(t=0, tmax=tmax)
