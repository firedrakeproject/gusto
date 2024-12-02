import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt
import os
import numpy as np

# file = 'Free_run/annular_vortex_mars_60-70_free_A0-0-norel_len-15sols_tracer_tophat'
file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--4-0_A0-0-norel_len-300sols_tracer_tophat-80'

lat_thresh = 72

path = f'/data/home/sh1293/results/{file}'

ds = xr.open_dataset(f'{path}/regrid_output.nc')
tracer = ds.tracer
integral_pole, latpole = fcs.tracer_integral(tracer, lat_thresh, 'pole')
integral_eq, lateq = fcs.tracer_integral(tracer, lat_thresh, 'equator')
integral_total = integral_eq + integral_pole

tracer['coslat'] = np.cos(tracer.lat * np.pi/180.)
integrand = tracer * tracer.coslat
true_total = integrand.integrate('lon').integrate('lat')

fig, ax = plt.subplots(1,1, figsize=(8,8))
integral_pole.plot(ax=ax, label=f'poleward {latpole}')
integral_eq.plot(ax=ax, label=f'equatorward {lateq}')
integral_total.plot(ax=ax, label='total')
true_total.plot(ax=ax, label='true total', linestyle='--')
plt.legend()

if not os.path.exists(f'{path}/Plots'):
    os.makedirs((f'{path}/Plots'))
plt.savefig(f'{path}/Plots/tracer_integral_{lat_thresh}_new.pdf')
print(f'Plot made:\n {path}/Plots/tracer_integral_{lat_thresh}_new.pdf')