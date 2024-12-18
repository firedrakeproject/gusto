import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt
import os
import numpy as np

# 'both', 'ann', 'free', 'will2'
run = 'both'

if run == 'both':
    file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-30sols_tracer_tophat-80_ref-4'
    t=450
    file1 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-30sols_tracer_tophat-80_ref-5'
    t1 = 112.5
    file2 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-30sols_tracer_tophat-80_ref-6'
    t2 = 112.5
    file3 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-30sols_tracer_tophat-80_ref-4_continuity'
    t3=450
elif run =='both_long':
    file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-30sols_tracer_tophat-80_ref-4'
    t=450
    file1 = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100sols_tracer_tophat-80_ref-6'
    t1=112.5
elif run == 'ann':
    file = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-300sols_tracer_tophat-80'
    file1 = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-3_PVpole--1-1_tau_r--2sol_A0-0-norel_len-30sols_tracer_tophat-80_ref-5'
elif run =='free':
    file = 'Free_run/annular_vortex_mars_60-70_free_A0-0-norel_len-30sols_tracer_tophat-80_ref-4'
    file1 = 'Free_run/annular_vortex_mars_60-70_free_A0-0-norel_len-30sols_tracer_tophat-80_ref-5'
elif run == 'will2':
    file = 'passive_tracer_williamson_2_tracer-coriolis'
    file1 = 'passive_tracer_williamson_2_tracer-coriolis_higher-res'


path = f'/data/home/sh1293/results/{file}'
path1 = f'/data/home/sh1293/results/{file1}'

# radius = 3396000
radius = 1

diag = xr.open_dataset(f'{path}/diagnostics.nc', group='tracer')
diag = diag.assign_coords(true_time=diag.time * t)

diag1 = xr.open_dataset(f'{path1}/diagnostics.nc', group='tracer')
diag1 = diag1.assign_coords(true_time=diag1.time * t1)

# diag = diag.where(diag.true_time<=np.max(diag1.true_time), drop=True)

fig, ax = plt.subplots(1,1, figsize=(8,8))
(diag['total']/radius**2).plot(ax=ax, x='true_time', label='diag total level 4')
(diag1['total']/radius**2).plot(ax=ax, x='true_time', label='diag total level 5')
try:
    path2 = f'/data/home/sh1293/results/{file2}'
    diag2 = xr.open_dataset(f'{path2}/diagnostics.nc', group='tracer')
    diag2 = diag2.assign_coords(true_time=diag2.time * t2)
    (diag2['total']/radius**2).plot(ax=ax, x='true_time', label='diag total level 6')
    path3 = f'/data/home/sh1293/results/{file3}'
    diag3 = xr.open_dataset(f'{path3}/diagnostics.nc',
    group='tracer')
    diag3 = diag3.assign_coords(true_time=diag3.time * t3)
    (diag3['total']/radius**2).plot(ax=ax, x='true_time', label='diag total continuity')
except:
    print('only two files')
plt.legend()

if not os.path.exists(f'{path}/Plots'):
    os.makedirs((f'{path}/Plots'))
plt.savefig(f'{path}/Plots/tracer_integral_lev_comp_cont.pdf')
print(f'Plot made:\n {path}/Plots/tracer_integral_lev_comp_cont.pdf')