import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import functions as fcs
import os
# from netCDF4 import Dataset
# from tomplot import (extract_gusto_field, extract_gusto_coords,             
#                         regrid_horizontal_slice)
import time
import pdb
from tqdm import tqdm

filepath = 'Bu2b4Rop2_l500dt250df30_n'
field_name = 'PotentialVorticity'
folder = 'PV'

plot_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots/{folder}'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

PV_structured, times = fcs.make_structured(filepath, field_name)
vmin = np.min(PV_structured[:,:,0])
vmax = np.max(PV_structured[:,:,0])

digits = len(str(len(times)))

# pdb.set_trace()


start_time = time.time()



for i in tqdm(range(len(times)), desc='Making plots'):
# for i in [0,1,2,3,4,5,6,7,8,9,10]:
    # print(f'{i:0{digits}d}')
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.set_aspect('equal')
    pcolor = PV_structured[:,:,i].plot.imshow(ax=ax, x='x', y='y', cmap='RdBu_r', extend='both', add_colorbar=False, vmin=vmin, vmax=vmax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{(times[i]/(2*np.pi/1.74e-4)):.3f} days')
    plot_name = f'{plot_dir}/pcolor_{i:0{digits}d}.pdf'
    # print(f'Saving figure to {plot_name}')
    plt.savefig(f'{plot_name}', bbox_inches='tight')
    plt.close()
print(f'File for gif script is:\n{plot_dir}')

end_time = time.time()

print(f'Total time taken {(end_time-start_time):.2f} seconds')