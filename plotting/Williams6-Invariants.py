"""
Plotting the invariants of the William 6 test case
"""
import matplotlib.pyplot as plt
import netCDF4 as nc


data = nc.Dataset("results/W6_SIQN_Ref=5_dt=500_diagnostics/diagnostics.nc")
time = data.variables['time'][:]
Enstrophy = data.groups['SWPotentialEnstrophy_from_PotentialVorticity']['total'][:] / data.groups['SWPotentialEnstrophy_from_PotentialVorticity']['l2'][0]

Kinetic_Energy = data.groups['ShallowWaterKineticEnergy']['l2'][:] / data.groups['ShallowWaterKineticEnergy']['l2'][0]
Potential_Energy = data.groups['ShallowWaterPotentialEnergy']['l2'][:] / data.groups['ShallowWaterPotentialEnergy']['l2'][0]

Divergence = data.groups['u_divergence']['l2'][:]
P_vorticity = data.groups['PotentialVorticity']['l2'][:] 
R_vorticity = data.groups['RelativeVorticity']['l2'][:] 
Total_Energy = (data.groups['ShallowWaterKineticEnergy']['l2'][:] + data.groups['ShallowWaterPotentialEnergy']['l2'][:]) / (data.groups['ShallowWaterKineticEnergy']['l2'][0] + data.groups['ShallowWaterPotentialEnergy']['l2'][0])



# Energy Plot
plt.ylabel('Energy (Joules')
plt.xlabel('Time (seconds)')
name = ['Kinetic Energy', 'Potential Energy', 'Total Energy (KE + V)']
for index, value in enumerate([Kinetic_Energy, Potential_Energy, Total_Energy]): 
    plt.plot(time, value)
    plt.title('%s Invariant' % name[index])
    plt.show()

# Vorticity plot 
plt.ylabel('Vorticity')
plt.xlabel('Time (seconds)')
name = ['Relative', 'Potential']
for index, value in enumerate([R_vorticity, P_vorticity]): 
    plt.plot(time, value)
    plt.title('%s Vorticity' % name[index])
    plt.show()

# Enstrophy Plt
plt.ylabel('Enstrophy')
plt.xlabel('Time')
plt.plot(time, Enstrophy)
plt.show()

# Divergence Plt
plt.ylabel('Divergence')
plt.xlabel('Time')
plt.plot(time, Divergence)
plt.show()