import functions as fcs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

filepath = 'new_single_flattrap_Bu2b1p5Rop2_l10dt250df1'

plot_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

pv, times = fcs.make_structured(filepath, 'PotentialVorticity')
maxpv = pv.max(dim='x').max(dim='y')
maxloc = pv.where(pv==maxpv, drop=True)
# minpv = pv.min(dim='x').min(dim='y')
xlist = maxloc.x
ylist = maxloc.y
polex = np.median(pv.x)
poley = np.median(pv.y)
polepv = pv.where(pv.x==polex, drop=True).where(pv.y==poley, drop=True).squeeze()

fig, ax = plt.subplots(1,1, figsize=(10,10/1.666))
maxpv_plot = maxpv.plot(ax=ax, label='Maximum PV')
polepv_plot = polepv.plot(ax=ax, label='Pole PV')#, linestyle=':')
# minpv_plot = minpv.plot(ax=ax, label='Minimum PV')
# ax.text(0.5, 0.5, f'Deviations from pole of max PV (m):\nx: {xlist.values-polex}\ny: {ylist.values-poley}', transform=ax.transAxes)
plt.legend()

plt.savefig(f'{plot_dir}/maxpv_evolution.pdf')
print(f'Plot made:\n{plot_dir}/maxpv_evolution.pdf')