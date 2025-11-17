from gusto import *
from firedrake import (IcosahedralSphereMesh, CellVolume)
import numpy as np

ref_lev = 5

# setup shallow water parameters
R = 3396000.    # Mars value (3389500)
H = 17000.      # Will's Mars value
Omega = 2*np.pi/88774.
g = 3.71

mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref_lev, degree=2)

### number of cells
V = FunctionSpace(mesh, "DG", 0)
f = Function(V)
print(f'Number of cells: {len(f.dat.data)}')

### size of cells
V = FunctionSpace(mesh, "DG", 0)
area = Function(V).interpolate(CellVolume(mesh))
print(f'Size of cells: {sqrt(area.dat.data.min()), sqrt(area.dat.data.max())}')