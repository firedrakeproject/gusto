import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat'

path = f'/data/home/sh1293/results/{file}'

sol_early = 200
sol_late = 300
lat = 72

if not os.path.exists(f'{path}/Plots/'):
    os.makedirs(f'{path}/Plots/')

ds = xr.open_dataset(f'{path}/regrid_output.nc')

ds['sol'] = ds.time/88774

fig, ax = plt.subplots(1, 1, figsize=(10,10))
ds_late = ds.where(ds.sol>=sol_early, drop=True).where(ds.sol<=sol_late, drop=True)
ds_plot = ds_late.where(ds_late.lat==lat, drop=True).mean(dim='lat')

cf = ds_plot.PotentialVorticity.plot.contourf(ax=ax, x='lon', y='time', cmap='OrRd', levels=21, add_colorbar=False)
cbar = plt.colorbar(cf, ax=ax, label='Potential Vorticity')
# cf1 = ds_plot.D_minus_H_rel_flag_less.plot.contourf(ax=ax, x='lon', y='time', colors='none', levels=[0, 0.5, 1.5], hatches=['', 'xx'])
cf1 = ds_plot.D_minus_H_rel_flag_less.plot.contour(ax=ax, x='lon', y='time', colors='green', levels=[0.5])
# add_colorbar_ax(ax, cf, cbar_label='PotentialVorticity', location='right', shrink=0.5)
# add_colorbar_ax(ax, cf1, cbar_label='co2_mask', location='left')
# artists, labels = cf1.legend_elements(str_format='{:2.1f}'.format)
# ax.legend(artists, labels, handleheight=2, framealpha=1)
ax.set_xlabel('longitude')
ax.set_ylabel('time')

sol_labels = ds_plot.sol.values
time_ticks = ds_plot.time.values
indices = np.linspace(0, len(time_ticks)-1, 4, dtype=int)
selected_time_ticks = time_ticks[indices]
selected_sol_labels = np.round(sol_labels[indices], 1)

ax.set_yticks(selected_time_ticks)
ax.set_yticklabels(selected_sol_labels)

plt.savefig(f'{path}/Plots/hovmoller.pdf')