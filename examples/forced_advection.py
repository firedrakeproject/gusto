from gusto import *
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh,
                       ExtrudedMesh, cos, sin, pi, exp, Function, Constant,
                       conditional)

# Implement forced advection test from Zerroukat and Allen 2020

nlayers = 50  # horizontal layers
columns = 200  # number of columns
L = 200e3
m = PeriodicIntervalMesh(columns, L)

# build volume mesh
H = 15e3  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

timestepping = TimesteppingParameters(dt=15)
dirname = 'forced_advection'
output = OutputParameters(dirname=dirname)
fieldlist = ["u", "rho", "theta"]

diagnostic_fields = [CourantNumber()]

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial conditions
u0 = state.fields("u")

# set up uexpr here
x, z = SpatialCoordinate(mesh)
u_mean = 10
uexpr = ((u_mean/2) * (2 + cos(2 * pi * x/L) * cos(pi * z) * sin(pi * z/H)),
         u_mean * H/L * sin(2 * pi * x/L) * sin(pi * z/H)**2)
u0.project(as_vector(uexpr))

# set up moisture variables here
theta0 = state.fields("theta")
Vth = theta0.function_space()
m1 = state.fields("m1", space=Vth)
m2 = state.fields("m2", space=Vth)
m3 = state.fields("m3", space=Vth)

x, z = SpatialCoordinate(state.mesh)
x = x/L # rescaled to domain
z = z/H

# set up constants and temperature and saturation profile
Gamma = Constant(-6.5e-3)   # lapse rate
H = Constant(15000)   # depth of mesh
T0 = Constant(293)   # temperature at surface
e1 = Constant(0.98) # level of saturation
e2 = Constant(2/3) # dimensionless height above which m1 is zero
T = Gamma * H * z + T0   # temperature profile 
ms = 3.8e-3 * exp((18 * T - 4824)/(T - 30))   # saturation profile

# initialise m1 as in Zerroukat and Allen 2020
m1.interpolate(conditional(1 > z/e2, 1 - z/e2, 0) * e1 * ms)

m1eqn = AdvectionEquation(state, Vth, equation_form="advective")
m2eqn = AdvectionEquation(state, Vth, equation_form="advective")
m3eqn = AdvectionEquation(state, Vth, equation_form="advective")

advected_fields = []
advected_fields.append(("m1", SSPRK3(state, m1, m1eqn)))
advected_fields.append(("m2", SSPRK3(state, m2, m2eqn)))
advected_fields.append(("m3", SSPRK3(state, m3, m3eqn)))

class Moisture(Physics):
    def __init__(self, state, ms):
        super().__init__(state)
        V = state.fields("m1").function_space()
        self.dm1 = Function(V)
        self.dm2 = Function(V)
        self.dm3 = Function(V)
        self.ms = ms

    def apply(self):
        ms = self.ms
        m1 = state.fields("m1")
        m2 = state.fields("m2")
        m3 = state.fields("m3")
        gamma1 = Constant(0.9)
        gamma2 = Constant(0.5)
        mr = Constant(1.e-4)
        dt = state.timestepping.dt
        self.dm1.interpolate(conditional(m1 - ms > 0, gamma1 * (m1 - ms), 0))
        self.dm2.interpolate(
            conditional(ms - m1 > 0,
                        conditional(ms - m1 < m2, gamma1 * (ms - m1), m2),
                        0))
        self.dm3.interpolate(conditional(m2 - mr > 0, gamma2 * (m2 - mr), 0))
        m1 += self.dm2 - self.dm1
        m2 += self.dm1 - self.dm2 - self.dm3
        m3 += self.dm3
        
moisture = Moisture(state, ms)

timestepper = AdvectionDiffusion(state, advected_fields,
                                 physics_list=[moisture])
timestepper.run(t=0, tmax=86400)
