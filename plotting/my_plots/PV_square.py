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
from mpl_toolkits.axes_grid1 import make_axes_locatable

def colourbar(mappable, extend):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(mappable, cax=cax, extend=extend)
    ax.set_aspect('equal', adjustable='box')
    return cb


filepath = 'new_single_flattrap_Bu2b1p5Rop2_l10dt250df1'
field_name = 'D_error'
folder = 'D_error'

plot_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}/Plots/{folder}'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

PV_structured, times = fcs.make_structured(filepath, field_name)

if field_name in ['PotentialVorticity', 'RelativeVorticity']:
    vmin = np.min(PV_structured[:,:,0])
    vmax = np.max(PV_structured[:,:,0])
elif field_name == 'D_error':
    But = filepath.split('Bu')[1].split('b')[0]
    try:
        Bui = float(But.split('p')[0])
        Bud = float(But.split('p')[1])*10**-len(But.split('p')[1])
        Bu = Bui+Bud
    except IndexError:
        Bu = float(But)
    g = 24.79
    Omega = 1.74e-4
    f0 = 2 * Omega        # Planetary vorticity
    rm = 1e6              # Radius of vortex (m)
    phi0 = Bu * (f0*rm)**2
    H = phi0/g
    PV_structured = PV_structured/H
    vmin = np.min(PV_structured[:,:,-1])
    vmax = np.max(PV_structured[:,:,-1])
    if abs(vmax)<abs(vmin):
        vmin=-vmax
    else:
        vmax=-vmin
# elif field_name == 'PotentialVorticity_error':
    # vmin = np.min(PV_structured[:,:,0])
    # vmax = np.max(PV_structured[:,:,0])

digits = len(str(len(times)))

# pdb.set_trace()


start_time = time.time()



for i in tqdm(range(len(times)), desc='Making plots'):
# for i in [0,1,2,3,4,5,6,7,8,9,10]:
    # print(f'{i:0{digits}d}')
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.set_aspect('equal')
    pcolor = PV_structured[:,:,i].plot.imshow(ax=ax, x='x', y='y', cmap='RdBu_r', extend='both', add_colorbar=False, vmin=vmin, vmax=vmax)
    cb = colourbar(pcolor, extend='both')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    if field_name=='D_error':
        ax.set_title(f'~{(vmax*100):.2f}% deviation from H\n{(times[i]/(2*np.pi/1.74e-4)):.3f} days')
    else:
        ax.set_title(f'{(times[i]/(2*np.pi/1.74e-4)):.3f} days')
    plot_name = f'{plot_dir}/pcolor_{i:0{digits}d}.pdf'
    # print(f'Saving figure to {plot_name}')
    plt.savefig(f'{plot_name}', bbox_inches='tight')
    plt.close()
print(f'File for gif script is:\n{plot_dir}')

end_time = time.time()

print(f'Total time taken {(end_time-start_time):.2f} seconds')