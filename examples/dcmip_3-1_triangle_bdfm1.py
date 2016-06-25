from __future__ import absolute_import
from gusto import *
from firedrake import *
nlayers = 10         # 10 horizontal layers
refinements = 2      # number of horizontal cells = 20*(4^refinements)

# build surface mesh
m = IcosahedralSphereMesh(radius=a, refinement_level=refinements)

# build volume mesh
z_top = 1.0e4        # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=z_top/nlayers,
                    extrusion_type="radial")

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="BDFM",
              dt=1.0,
              g=9.81)

# interpolate initial conditions
state.initialise_state_from_expressions(u_expr, rho_expr, theta_expr)

# build time stepper
stepper = Timestepper(state, advection_list)

stepper.run(t=0, dt=1.25, T=3600.0)
