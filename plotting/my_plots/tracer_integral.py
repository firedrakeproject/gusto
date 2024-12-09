import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt
import os
import numpy as np

# file = 'Free_run/annular_vortex_mars_60-70_free_A0-0-norel_len-30sols_tracer_tophat-80_ref-4'
# file = 'Free_run/annular_vortex_mars_60-70_free_A0-0-norel_len-30sols_tracer_tophat-80_ref-5'
file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--4-0_A0-0-norel_len-300sols_tracer_tophat-80'
# file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-30sols_tracer_tophat-80_ref-5'
# file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-30sols_tracer_tophat-80_ref-6'
# file = 'Free_run/annular_vortex_mars_60-70_free_A0-0-norel_len-30sols_tracer_tophat-80_ref-4_tracer100'
# file = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-3_PVpole--1-1_tau_r--2sol_A0-0-norel_len-30sols_tracer_tophat-80_ref-5'
# file = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-300sols_tracer_tophat-80'
# file = 'passive_tracer_williamson_2_tracer-coriolis'
# file = 'passive_tracer_williamson_2_tracer-coriolis_higher-res'
# file = 'passive_tracer_williamson_2_tracer-coriolis_long'
# file = 'passive_tracer_williamson_2_tracer-gaussian'
# file = 'passive_tracer_williamson_2_tracer-gaussian_long'
# file = 'passive_tracer_williamson_2_tracer-tophat'
# file = 'passive_tracer_williamson_2_tracer-tophat_long'

norm = True

lat_thresh = 72

path = f'/data/home/sh1293/results/{file}'

# radius = 3396000
radius = 1

# ds = xr.open_dataset(f'{path}/regrid_output.nc')
# tracer = ds.tracer
# integral_pole, latpole = fcs.tracer_integral(tracer, lat_thresh, 'pole')
# integral_eq, lateq = fcs.tracer_integral(tracer, lat_thresh, 'equator')
# integral_total = integral_eq + integral_pole

# tracer['coslat'] = np.cos(tracer.lat * np.pi/180.)
# integrand = tracer * tracer.coslat
# integrand['lat'] = integrand.lat * np.pi/180.
# integrand['lon'] = integrand.lon * np.pi/180.
# true_total = integrand.integrate('lon').integrate('lat')

diag = xr.open_dataset(f'{path}/diagnostics.nc', group='tracer')
diag = diag.assign_coords(true_time=diag.time * 450)


fig, ax = plt.subplots(1,1, figsize=(8,8))
# integral_pole.plot(ax=ax, label=f'poleward {latpole}')
# integral_eq.plot(ax=ax, label=f'equatorward {lateq}')
# integral_total.plot(ax=ax, label='total')
# true_total.plot(ax=ax, label='true total', linestyle='--')
if not norm:
    (diag['total']/radius**2).plot(ax=ax, x='true_time', label='diag total')
elif norm:
    (diag['total']/diag['total'].values[0]-1).plot(ax=ax, x='true_time', label='diag total (normalised-1)')
plt.legend()

if not os.path.exists(f'{path}/Plots'):
    os.makedirs((f'{path}/Plots'))
plt.savefig(f'{path}/Plots/tracer_integral_{lat_thresh}_new.pdf')
print(f'Plot made:\n {path}/Plots/tracer_integral_{lat_thresh}_new.pdf')