import xarray as xr
import functions as fcs
import matplotlib.pyplot as plt
import os
import numpy as np

file = 'Free_run/annular_vortex_mars_60-70_free_A0-0-norel_len-30sols_tracer_tophat-80_ref-4'
file1 = 'Free_run/annular_vortex_mars_60-70_free_A0-0-norel_len-30sols_tracer_tophat-80_ref-4_tracer100'

path = f'/data/home/sh1293/results/{file}'
path1 = f'/data/home/sh1293/results/{file1}'

radius = 3396000

diag = xr.open_dataset(f'{path}/diagnostics.nc', group='tracer')
diag = diag.assign_coords(true_time=diag.time * 450)

diag1 = xr.open_dataset(f'{path1}/diagnostics.nc', group='tracer')
diag1 = diag1.assign_coords(true_time=diag1.time * 450)

diag1 = diag1.where(diag1.true_time<=np.max(diag.true_time), drop=True)
diag1 = diag1/100

fig, ax = plt.subplots(1,1, figsize=(8,8))
(diag['total']/radius**2).plot(ax=ax, x='true_time', label='diag total tracer=1')
(diag1['total']/radius**2).plot(ax=ax, x='true_time', label='diag total tracer=100')
plt.legend()

if not os.path.exists(f'{path}/Plots'):
    os.makedirs((f'{path}/Plots'))
plt.savefig(f'{path}/Plots/tracer_integral_mag_comp.pdf')
print(f'Plot made:\n {path}/Plots/tracer_integral_mag_comp.pdf')