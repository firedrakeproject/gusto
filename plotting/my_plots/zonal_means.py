import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

results_dir = f'/data/home/sh1293/results/Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_alpha--1_working_long'

time = 0

ds = xr.open_dataset(f'{results_dir}/regrid_output.nc')

ds_zonalmean = ds.mean('lon')
ds_zonalmean_init = ds_zonalmean.loc[dict(time=time)]

dticks = np.array([-90, -60, -30, 0, 30, 60, 90])
rticks = dticks * np.pi/180

omega = 2*np.pi/88774
hbart = 17000

qticks1 = [-2*omega/hbart, 0, 2*omega/hbart]
qticks2 = [r'$-\dfrac{2\Omega}{H}$', r'$0$', r'$\dfrac{2\Omega}{H}$']

rlat = ds_zonalmean_init.lat
q = ds_zonalmean_init.PotentialVorticity
h = ds_zonalmean_init.D
u=ds_zonalmean_init.u_zonal

fig, axs = plt.subplots(3, 1, sharex=True, figsize = (6,9))
axs[0].plot(rlat, q, color = 'blue')
axs[0].set_ylabel(r'PV: $q[m^{-1}s^{-1}]$')
axs[0].set_yticks(qticks1, qticks2)
axs[0].plot(rlat, np.sin(rlat*np.pi/180)*2*omega/hbart, '--', color='black', alpha=0.5)
# q.plot(ax=axs[0])
axs[1].plot(rlat, [0]*len(rlat), '--', color = 'black', alpha = 0.5)
axs[1].plot(rlat, h/hbart-1, color = 'blue')
axs[1].set_ylabel(r'height perturbation: $\dfrac{h}{H}-1$')
axs[2].plot(rlat, [0]*len(rlat), '--', color = 'black', alpha = 0.5)
axs[2].plot(rlat, u, color = 'blue')
axs[2].set_ylabel(r'zonal speed: $u\,[ms^{-1}]$')
axs[2].set_xlabel(r'latitude: $\phi$')
# axs[2].set_xticks(rticks, dticks)
fig.tight_layout()

plt.savefig(f'{results_dir}/{time}_from_results.pdf')