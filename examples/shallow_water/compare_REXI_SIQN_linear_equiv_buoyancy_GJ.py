from vtk import vtkXMLUnstructuredGridReader
import vtk.util.numpy_support as vnp
from numpy import linalg
import netCDF4 as nc
import matplotlib.pyplot as plt

# -------------------------------------------------------------------- #
# REXI solution error
# -------------------------------------------------------------------- #

# initial conditions
ICs_file = "/Users/Jemma/firedrake/src/gusto/examples/shallow_water/pileus_results/rexi_linear_equivalent_buoyancy_galewsky_jet_0.vtu"

# read initial conditions
reader1 = vtkXMLUnstructuredGridReader()
reader1.SetFileName(ICs_file)
reader1.Update()  # Needed because of GetScalarRange
ICs = reader1.GetOutput()

# results
results_file = "/Users/Jemma/firedrake/src/gusto/examples/shallow_water/pileus_results/rexi_linear_equivalent_buoyancy_galewsky_jet_1.vtu"

# read results
reader2 = vtkXMLUnstructuredGridReader()
reader2.SetFileName(results_file)
reader2.Update()  # Needed because of GetScalarRange
results = reader2.GetOutput()

# arrays
u0_array = ICs.GetPointData().GetArray("input[0]")
D0_array = ICs.GetPointData().GetArray("input[1]")
b0_array = ICs.GetPointData().GetArray("input[2]")
q0_array = ICs.GetPointData().GetArray("input[3]")

u_array = results.GetPointData().GetArray("input[0]")
D_array = results.GetPointData().GetArray("input[1]")
b_array = results.GetPointData().GetArray("input[2]")
q_array = results.GetPointData().GetArray("input[3]")

# convert to numpy arrays
np_u0 = vnp.vtk_to_numpy(u0_array)
np_u = vnp.vtk_to_numpy(u_array)
np_D0 = vnp.vtk_to_numpy(D0_array)
np_D = vnp.vtk_to_numpy(D_array)
np_b0 = vnp.vtk_to_numpy(b0_array)
np_b = vnp.vtk_to_numpy(b_array)
np_q0 = vnp.vtk_to_numpy(q0_array)
np_q = vnp.vtk_to_numpy(q_array)

# error arrays
u_error = np_u0 - np_u
D_error = np_D0 - np_D
b_error = np_b0 - np_b
q_error = np_q0 - np_q

# l2 norms of the error
u_error_l2 = linalg.norm(u_error)
D_error_l2 = linalg.norm(D_error)
b_error_l2 = linalg.norm(b_error)
q_error_l2 = linalg.norm(q_error)

# l2 norms of the field
u_l2 = linalg.norm(np_u)
D_l2 = linalg.norm(np_D)
b_l2 = linalg.norm(np_b)
q_l2 = linalg.norm(np_q)

# normalise the error norm by dividing by the field norm
u_norm = u_error_l2/u_l2
D_norm = D_error_l2/D_l2
b_norm = b_error_l2/b_l2
q_norm = q_error_l2/q_l2

print(u_norm)
print(D_norm)
print(b_norm)
print(q_norm)

# -------------------------------------------------------------------- #
# SIQN solution error
# -------------------------------------------------------------------- #

SIQN_file = "/Users/Jemma/firedrake/src/gusto/examples/shallow_water/results/linear_equivalent_buoyancy_galewsky_jet_SIQN/diagnostics.nc"

SIQN_data = nc.Dataset(SIQN_file)

# error
SIQN_u_error = SIQN_data.groups['u_error']
SIQN_D_error = SIQN_data.groups['D_error']
SIQN_b_error = SIQN_data.groups['b_e_error']
SIQN_q_error = SIQN_data.groups['q_t_error']

# fields (for normalising)
SIQN_u = SIQN_data.groups['u']
SIQN_D = SIQN_data.groups['D']
SIQN_b = SIQN_data.groups['b_e']
SIQN_q = SIQN_data.groups['q_t']

# l2 norm of the error
SIQN_u_error_l2 = SIQN_u_error['l2'][:]
SIQN_D_error_l2 = SIQN_D_error['l2'][:]
SIQN_b_error_l2 = SIQN_b_error['l2'][:]
SIQN_q_error_l2 = SIQN_q_error['l2'][:]

# l2 norm of the field
SIQN_u_l2 = SIQN_u['l2']
SIQN_D_l2 = SIQN_D['l2']
SIQN_b_l2 = SIQN_b['l2']
SIQN_q_l2 = SIQN_q['l2']

# normalise the error norm by dividing by the field norm
SIQN_u_norm = SIQN_u_error_l2[-1]/SIQN_u_l2[-1]
SIQN_D_norm = SIQN_D_error_l2[-1]/SIQN_D_l2[-1]
SIQN_b_norm = SIQN_b_error_l2[-1]/SIQN_b_l2[-1]
SIQN_q_norm = SIQN_q_error_l2[-1]/SIQN_q_l2[-1]

print(SIQN_u_norm)
print(SIQN_D_norm)
print(SIQN_b_norm)
print(SIQN_q_norm)

# -------------------------------------------------------------------- #
# plot errors
# -------------------------------------------------------------------- #
x_axis = ['u', 'D', 'b_e', 'q_t']
rexi_errors = [u_norm, D_norm, b_norm, q_norm]
SIQN_errors = [SIQN_u_norm, SIQN_D_norm, SIQN_b_norm, SIQN_q_norm]

plt.scatter(x_axis, rexi_errors, marker='o', s=70, label="REXI")
plt.scatter(x_axis, SIQN_errors, marker='v', s=70, label="SIQN")
plt.legend()
plt.xlabel("field name")
plt.ylabel("normalised L2 error")
plt.show()
