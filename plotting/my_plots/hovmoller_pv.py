import functions as fcs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

filepath = 'new_single_fplane_Bu2b1p5Rop2_l500dt250df30'

plot_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

pv, times = fcs.make_structured(filepath, 'PotentialVorticity')
spacing = pv.x[-1]/len(pv.x)
ypole = np.median(pv.y)
pvcs = pv.where(pv.y==ypole, drop=True).squeeze()
xpole = np.median(pv.x)
pvcs_centre = pvcs.where(pv.x>=xpole-40*spacing, drop=True).where(pv.x<=xpole+40*spacing, drop=True)


fig, ax = plt.subplots(1,1, figsize=(10,20))
pcolor = pvcs_centre.plot.imshow(ax=ax, x='x', y='time', cmap='RdBu_r', origin='upper')

plt.savefig(f'{plot_dir}/hovmoller_pv.pdf')
print(f'Plot made:\n{plot_dir}/hovmoller_pv.pdf')