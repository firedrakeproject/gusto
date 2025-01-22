import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt
import os
import numpy as np


# file = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-300sols_tracer_tophat-80_ref-4'
file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat-80_ref-5'

ref_lev = 5

fix_lat_thresh = False

norm = False

lat_thresh = 60

path = f'/data/home/sh1293/results/{file}'

radius = 3396000
# radius = 1

ds = xr.open_dataset(f'{path}/regrid_output.nc')
q = ds.PotentialVorticity
max_zonal_mean = fcs.max_zonal_mean(q)
tracer = ds.tracer

q['sol'] = q.time/88774
ds_late = q.where(q.sol>=100, drop=True).where(q.sol<=300, drop=True)
if not fix_lat_thresh:
    lat_thresh_time = fcs.max_zonal_mean(ds_late)
    lat_thresh = lat_thresh_time.max_lat.mean(dim='time').values
# tracer['lat_thresh'] = max_zonal_mean.max_lat
# integral_pole, latpole = fcs.tracer_integral(tracer, max_zonal_mean.max_lat, 'pole')
# integral_eq, lateq = fcs.tracer_integral(tracer, max_zonal_mean.max_lat, 'equator')
integral_pole, latpole = fcs.tracer_integral(tracer, lat_thresh, 'pole')
integral_eq, lateq = fcs.tracer_integral(tracer, lat_thresh, 'equator')
integral_total = integral_eq + integral_pole

tracer['coslat'] = np.cos(tracer.lat * np.pi/180.)
integrand = tracer * tracer.coslat
integrand['lat'] = integrand.lat * np.pi/180.
integrand['lon'] = integrand.lon * np.pi/180.
true_total = integrand.integrate('lon').integrate('lat')

diag = xr.open_dataset(f'{path}/diagnostics.nc', group='tracer')

dt = (0.5)**(ref_lev-4) * 450.

diag = diag.assign_coords(true_time=diag.time * dt)


fig, ax = plt.subplots(1,1, figsize=(8,8))
integral_pole.plot(ax=ax, label=f'poleward {latpole}')
integral_eq.plot(ax=ax, label=f'equatorward {lateq}')
integral_total.plot(ax=ax, label='total')
true_total.plot(ax=ax, label='true total', linestyle='--')
if not norm:
    (diag['total']/radius**2).plot(ax=ax, x='true_time', label='diag total')
elif norm:
    (diag['total']/diag['total'].values[0]-1).plot(ax=ax, x='true_time', label='diag total (normalised-1)')
# ax.set_yscale('log')
plt.legend()

if not os.path.exists(f'{path}/Plots'):
    os.makedirs((f'{path}/Plots'))
plt.savefig(f'{path}/Plots/tracer_integral_{lat_thresh}.pdf')
print(f'Plot made:\n {path}/Plots/tracer_integral_{lat_thresh}.pdf')

# plt.savefig(f'{path}/Plots/tracer_integral_var-lat.pdf')
# print(f'Plot made:/n {path}/Plots/tracer_integral_var-lat.pdf')