import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import functions as fcs
import xrft
import os

file = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-300sols_tracer_tophat-80_ref-4'

max_wavnum = 10

path = f'/data/home/sh1293/results/{file}'


ds = xr.open_dataset(f'{path}/regrid_output.nc')
q = ds.PotentialVorticity

q_late = q.where(q.time>=100*88774., drop=True)
lat_val = fcs.max_zonal_mean(q_late).max_lat.mean(dim='time').values

fft = fcs.fft(q, lat_val, 10, max_wavnum)

cmap = plt.get_cmap("coolwarm_r")

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_ylim(0, 5.5e-7)
for i in range(1,max_wavnum+1):
    fft[:,i].plot(ax=ax, x='time', label=f'{i}', color=cmap(i/max_wavnum))

plt.legend()

plt.savefig(f'{path}/Plots/zonal_fft.pdf')
print(f'Plot made:\n {path}/Plots/zonal_fft.pdf'