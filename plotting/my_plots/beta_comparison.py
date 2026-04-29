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
import sys
from matplotlib.colors import LogNorm, Normalize

def colourbar(mappable, extend):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(mappable, cax=cax, extend=extend)
    ax.set_aspect('equal', adjustable='box')
    return cb


# jupiter_sw or vp20_moist_jupiter
folder = 'jupiter_sw'
long = False
rad = True
Bu10 = True

if folder == 'vp20_moist_jupiter':
    file1 = 'single-step_trap-qg_tophat_cD1em3gamma3900q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file2 = 'single-step_trap-qg_tophat_cD1em3gamma39000q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file3 = 'single-step_trap-qg_tophat_cD1em3gamma390000q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file1_rad = 'single-step_trap-qg_tophat_radt5cD1em3gamma3900q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file2_rad = 'single-step_trap-qg_tophat_radt5cD1em3gamma39000q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file3_rad = 'single-step_trap-qg_tophat_radt5cD1em3gamma390000q05em1xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file1_long = 'single-step_trap-qg_tophat_radt5cD1em3gamma3900q05em1xi1em1_Bu1b1p5Rop2_l200dt250df30'
    file2_long = 'single-step_trap-qg_tophat_radt5cD1em3gamma39000q05em1xi1em1_Bu1b1p5Rop2_l200dt250df30'
    file3_long = 'single-step_trap-qg_tophat_radt5cD1em3gamma390000q05em1xi1em1_Bu1b1p5Rop2_l200dt250df30'

elif folder == 'jupiter_sw':
    file1 = 'single-step_trap_beta3900q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file2 = 'single-step_trap_beta39000q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file3 = 'single-step_trap_beta390000q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file1_rad = 'single-step_trap_radt5beta3900q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file2_rad = 'single-step_trap_radt5beta39000q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file3_rad = 'single-step_trap_radt5beta390000q01em2xi1em1_Bu1b1p5Rop2_l10dt250df10'
    file1_Bu10 = 'single-step_trap_radt5beta3900q01em2xi1em1_Bu10b1p5Rop2_l10dt250df10'
    file2_Bu10 = 'single-step_trap_radt5beta39000q01em2xi1em1_Bu10b1p5Rop2_l10dt250df10'
    file3_Bu10 = 'single-step_trap_radt5beta390000q01em2xi1em1_Bu10b1p5Rop2_l10dt250df10'

files = [file1, file1, file2, file3]
if long:
    files = [file1_long, file1_long, file2_long, file3_long]
if rad:
    files = [file1_rad, file1_rad, file2_rad, file3_rad]
if Bu10:
    files = [file1_Bu10, file1_Bu10, file2_Bu10, file3_Bu10]
    rad = True
    long = False
names = ['Initial condition', rf'$\beta_1=3900$', rf'$\beta_1=39000$', rf'$\beta_1=390000$']


field_name = 'RelativeVorticity'
extend = 'both'

Omega = 1.74e-4
f0 = 2*Omega
Bu = 1
rm = 1e6
g = 24.79
phi0 = Bu*(f0*rm)**2
H = phi0/g

fig, axs = plt.subplots(2,2, figsize=(8,8))

for i in range(4):
    filepath = files[i]
    plot_dir = f'/data/home/sh1293/results/{folder}/{filepath}/Plots/{field_name}'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    PV_structured, times = fcs.make_structured(filepath, field_name, folder=folder)

    # breakpoint()

    PV_structured = PV_structured[196:315, 197:316, :]


    cmap = 'RdBu_r'
    if i==0:
        # vmin = np.min(PV_structured[:,:,0])
        vmin = 0# if not long else 1e-10
        # vmax = np.max(PV_structured[:,:,0])
        # vmax = 0.00025
        vmax = 0.75*f0
        # vmax = np.max(PV_structured[:,:,-1])
    # vmax = f0/H

    # norm = LogNorm(vmin=vmin, vmax=vmax) if long else Normalize(vmin=vmin, vmax=vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)

    axs[i//2][i%2].set_box_aspect(1)
    if i==0:
        pcolor = PV_structured[:,:,0].plot.imshow(ax=axs[i//2][i%2], x='x', y='y', cmap=cmap, extend=extend, add_colorbar=False, vmin=vmin, vmax=vmax, norm=norm)
    else:
        pcolor = PV_structured[:,:,-1].plot.imshow(ax=axs[i//2][i%2], x='x', y='y', cmap=cmap, extend=extend, add_colorbar=False, vmin=vmin, vmax=vmax, norm=norm)

    axs[i//2][i%2].set_xlabel('')
    axs[i//2][i%2].set_ylabel('')
    axs[i//2][i%2].set_xticks([])
    axs[i//2][i%2].set_yticks([])
    axs[i//2][i%2].set_title(f'{names[i]}')

# pos = axs[0,0].get_position()
cbar_ax = fig.add_axes([
    0.25,
    0.05,
    0.5,
    0.03
])
cb = fig.colorbar(pcolor, cax=cbar_ax, orientation='horizontal', extend=extend)#, ticks=([0,0.25*f0, 0.5*f0, 0.75*f0]))
# cb.set_ticklabels([r'$0$', r'$f_0/4$', r'$f_0/2$', r'$3f_0/4$'])
cb.set_label('Relative Vorticity')

# print(f'Saving figure to {plot_name}')
if long:
    extra_name = '_long'
elif rad:
    extra_name = '_rad'
else:
    extra_name = ''
if Bu10:
    extra_name = f'{extra_name}_Bu10'
plt.savefig(f'/data/home/sh1293/results/{folder}/beta_comparison{extra_name}.pdf', bbox_inches='tight')
plt.close()
print(f'Plot made:\n/data/home/sh1293/results/{folder}/beta_comparison{extra_name}.pdf')