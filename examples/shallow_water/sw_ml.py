from firedrake import *
from gusto import *

Lx = Ly = 100
nx = ny = 10
mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction="both", name="mesh")
dt = 0.02

g = 9.80616
H = 3e4/g
parameters = ShallowWaterParameters(mesh, H=H, g=g,
                                    rotation=CoriolisOptions.nonrotating)

eqns = ShallowWaterEquations

output = OutputParameters(dirname="sw_ml", dump_vtus=True)
sw_model = GustoModel(mesh, dt, parameters, eqns, family='BDM')
sw_model.setup(output)

ml_model = PointNN(n_in=5, n_out=3)

hybrid_model = HybridModel(sw_model, ml_model, fields_to_adjust=["u", "D"])

hybrid_model.generate_data(filename)

hybrid_model.train()

hybrid_model.evaluate()
