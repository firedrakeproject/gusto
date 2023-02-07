from gusto import *
from firedrake import (PeriodicRectangleMesh, exp, Constant, sqrt, cos)

# mesh depends on parameters so set these up first
H = 10000
R = 6371220
parameters = ShallowWaterParameters(H=H)
g = parameters.g
Omega = parameters.Omega
beta = 2*Omega/R
Ld = ((g*H)**1/4)/(beta**(1/2))
