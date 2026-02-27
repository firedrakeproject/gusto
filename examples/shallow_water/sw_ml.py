from firedrake import *
from gusto import *
from numpy import random

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
subcycling_options = SubcyclingOptions(subcycle_by_courant=0.25)
sw_model.setup(output, subcycling_options=subcycling_options)
Vdg = sw_model.stepper.fields("D").function_space()
Dbar = Function(Vdg).assign(H)
sw_model.stepper.set_reference_profiles([("D", Dbar)])

ml_model = PointNN(n_in=5, n_out=3)

hybrid_model = HybridModel(sw_model, ml_model,
                           input_fields=["u", "D"],
                           fields_to_adjust=["u", "D"],
                           data_dir="results/sw_ml")

def generate_initial_conditions(mesh, n_ics, n_gaussians, scale, extent, H):
    """
    Args:
        mesh (:class:`Mesh`): the model's mesh.
        n_ics (int): the number of different initial conditions to generate
            (each corresponding to a different simulation).
        n_gaussians (int): the number of random Gaussians to generate in each
            initial condition.
        scale (float): a scaling factor that controls the placement of the
            Gaussians, related to the size of the mesh.
        extent (float): a scaling factor that controls the width of the
            Gaussians.
    """
    ics = []
    x, y = SpatialCoordinate(mesh)
    # Produce n random samples for initial conditions
    for r in range(n_ics):
        ic = H
        for i in range(n_gaussians):
            x_pos = random.uniform(low=0.2, high=0.8) * scale
            y_pos = random.uniform(low=0.2, high=0.8) * scale
            a = 1 + random.rand()
            gaussian = a * exp(-((x-x_pos)**2)/extent-((y-y_pos)**2)/extent)
            ic += H * gaussian
        ics.append([("D", ic)])
    return ics

ics = generate_initial_conditions(mesh, 10, 6, Lx, Lx, H)
#hybrid_model.generate_data(ics, ["u", "D"], ndt=4)
dir_list = ["results/sw_ml/test_train_0", "results/sw_ml/test_train_1",
            "results/sw_ml/test_train_2", "results/sw_ml/test_train_3",
            "results/sw_ml/test_train_4", "results/sw_ml/test_train_5",
            "results/sw_ml/test_train_6", "results/sw_ml/test_train_7",
            "results/sw_ml/test_train_8", "results/sw_ml/test_train_9"]
#hybrid_model.process_data(["u", "D"], ndt=4, dir_list=dir_list)

hybrid_model.train("point_train_data.npy", "global_train_data.h5",
                   "point_test_data.npy", "global_test_data.h5")

hybrid_model.evaluate()
