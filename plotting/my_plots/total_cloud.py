import functions as fcs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

filepath = 'beta3900q01em3xi1em2_Bu1b1p5Rop2_l200dt250df30'

plot_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

cloud, times = fcs.make_structured(filepath, 'cloud_water')
total_cloud = (cloud).integrate('x').integrate('y')

fig, ax = plt.subplots(1,1, figsize=(10,10/1.666))
total_plot = total_cloud.plot(ax=ax, label='Total amount of cloud water')
plt.legend()

plt.savefig(f'{plot_dir}/totalcloud_evolution.pdf')
print(f'Plot made:\n{plot_dir}/totalcloud_evolution.pdf')