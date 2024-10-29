import os
import matplotlib.pyplot as plt
import xarray as xr

file = 'Relax_to_annulus/annular_vortex_mars_55-60_PVmax--2-4_PVpole--1-0_tau_r--2sol_A0-0.0-norel_len-300sols_tracer_tophat-80'

sol_early = 100
sol_late = 300

path = f'/data/home/sh1293/results/{file}'

if not os.path.exists(f'{path}/Plots/'):
    os.makedirs(f'{path}/Plots/')

ds = xr.open_dataset(f'{path}/regrid_output.nc')

ds['sol'] = ds.time/88774

ds_late = ds.where(ds.sol>=sol_early, drop=True).where(ds.sol<=sol_late, drop=True)

ds_late_time_mean = ds_late.mean(dim='time')

ds_late_time_zonal_mean = ds_late_time_mean.mean(dim='lon')

fig, ax = plt.subplots(1, 1, figsize=(8,8))

plot = ds_late_time_zonal_mean.PotentialVorticity.plot(ax=ax)

plt.savefig(f'{path}/Plots/late_time_pv_profile_{sol_early}-{sol_late}sol.pdf')