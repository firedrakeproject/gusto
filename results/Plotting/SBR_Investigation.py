import netCDF4 as nc 
import matplotlib.pyplot as plt
from numpy import reshape
fp="results/Invar_Manifold/results/SBR_invariant_solve/diagnostics.nc"
data =nc.Dataset(fp)
time = data.variables['time'][:]
u_merid_min = data.groups['u_meridional']['min'][:]
u_merid_max = data.groups['u_meridional']['max'][:]
u_merid_total = data.groups['u_meridional']['total'][:]
u_merid_rms = data.groups['u_meridional']['total'][:]
u_merid_l2 = data.groups['u_meridional']['l2'][:]
u_merid_l2_norm = data.groups['u_meridional']['l2'][:] / data.groups['u_meridional']['l2'][0]
u_merid_measure = [[u_merid_min], [u_merid_max], [u_merid_l2], [u_merid_l2_norm], [u_merid_rms], [u_merid_total]]
measures = ['min', 'max', 'l2', 'l2_norm', 'rms', 'total']
rows , cols = 3, 2
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(rows, cols,
                        sharex = 'col')
mes = 0
for row in range(rows):
    for col in range(cols):
        ax[row,col].plot(time, reshape(u_merid_measure[mes], (148,1)))
        ax[row,col].set(title=f"Graph of Meridonal velocity, {measures[mes]} measure",
                        ylabel = measures[mes])
        mes += 1
plt.show()

