import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import functions as fcs
import pdb
import os

file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4'

path = '/data/home/sh1293/results'

if not os.path.exists(f'{path}/{file}/Plots/'):
    os.makedirs(f'{path}/{file}/Plots/')

ds = xr.open_dataset(f'{path}/{file}/regrid_output.nc')

q = ds.PotentialVorticity
t = ds.tracer_rs

t100 = 100 * 88774
ind100 = int((np.ceil(t100/4500)+1))

q_late = q.where(q.time>=100*88774., drop=True).mean(dim='time').mean(dim='lon')
t_late = t.where(t.time>=270*88774., drop=True).mean(dim='time').mean(dim='lon')
t_initial = t[ind100,:,:].mean(dim='lon')

fig, ax1 = plt.subplots(1,1, figsize=(8,8))
ax2 = ax1.twinx()
q_late.plot(ax=ax1, color='red', label='PV')
t_late.plot(ax=ax2, color='blue', label='tracer')
t_initial.plot(ax=ax2, color='blue', alpha=0.5, linestyle='--', label='initial tracer')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.savefig(f'{path}/{file}/Plots/pv-tracer.pdf')
print(f'Plot made:\n {path}/{file}/Plots/pv-tracer.pdf')