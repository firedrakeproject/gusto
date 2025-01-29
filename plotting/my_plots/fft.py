import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import functions as fcs
import xrft
import os

# file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat-80_ref-5'
file = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-300sols_tracer_tophat-80_ref-4'

max_wavnum = 10
lat_range = 10

path = f'/data/home/sh1293/results/{file}'


ds = xr.open_dataset(f'{path}/regrid_output.nc')
q = ds.PotentialVorticity

q_late = q.where(q.time>=100*88774., drop=True)
lat_val = fcs.max_zonal_mean(q_late).max_lat.mean(dim='time').values

fft = fcs.fft(q, lat_val, lat_range, max_wavnum)
fft_ave = fft.where(fft.time>=100*88774., drop=True).mean(dim='time')
fft_ave_cut = fft_ave.where(fft_ave.freq_lon>0, drop=True)

cmap = plt.get_cmap("coolwarm_r")

fig, axs = plt.subplots(2,1, figsize=(8,12))
axs[0].set_ylim(0, 5.5e-7)
axs[1].set_ylim(0, 1.1e-7)
for i in range(1,max_wavnum+1):
    fft[:,i].plot(ax=axs[0], x='time', label=f'{i}', color=cmap(i/max_wavnum))
axs[1].bar(x=fft_ave_cut.freq_lon, height=fft_ave_cut.values)
axs[1].set_xlabel('wavenumber')
axs[1].set_ylabel('power (100-300sol ave)')

axs[0].legend()

plt.savefig(f'{path}/Plots/zonal_fft.pdf')
print(f'Plot made:\n {path}/Plots/zonal_fft.pdf')


