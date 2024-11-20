import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
import matplotlib.path as mpath
from matplotlib import (cm, colors, gridspec)
import matplotlib.ticker as mticker
import pdb
import os

file = 'Free_run/annular_vortex_mars_60-70_free_A0-0-norel_len-15sols_tracer_tophat'

start_sol = 100

results_dir = f'/data/home/sh1293/results/{file}'

if not os.path.exists(f'{results_dir}/Plots/Tracer_stereo/'):
    os.makedirs(f'{results_dir}/Plots/Tracer_stereo/')

# timeind = -1

ds = xr.open_dataset(f'{results_dir}/regrid_output.nc')
q = ds.tracer


theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)



for timeind in range(0, len(q.time)):
# for timeind in [275, 295]:
# for timeind in range(0, 10):
    # if timeind%100==0:
    print(timeind)
    fig = plt.figure(figsize = (8, 8))
    spec = gridspec.GridSpec(ncols=1, nrows=1, width_ratios=[1], figure=fig)
    ax = fig.add_subplot(spec[0], projection = ccrs.NorthPolarStereo())
    gl = ax.gridlines(crs = ccrs.PlateCarree(), linewidth = 1, linestyle = '--', color = 'black', alpha = 1, draw_labels=False)
    meridians = [0, 60, 120, 180, -60, -120]
    parallels = [50, 60, 70, 80]
    gl.xlocator = mticker.FixedLocator(meridians)
    gl.ylocator = mticker.FixedLocator(parallels)
    gl.xlabels = False
    #gl.ylabels = [True if parallel in [50] else False]
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-180,180,50,90], crs=ccrs.PlateCarree())
    # pdb.set_trace()
    # contourplot = q[timeind,:,:].plot.contourf(ax=ax, vmin = 0, vmax = 1.75*val, add_colorbar=True,
    #                             transform = ccrs.PlateCarree(), cmap='OrRd', levels=np.linspace(0, 1.75*val, 15), extend = 'both')

    try:
        contourplot = q[timeind,:,:].plot.contourf(ax=ax, add_colorbar=True,
                                transform = ccrs.PlateCarree(), cmap='OrRd', levels=20, extend = 'both')



        colorbar = contourplot.colorbar
        colorbar.set_label(r'Tracer')

        ax.set_title(f'{ds.time.values[timeind]/88774:.4f}sol')

        plt.savefig(f'{results_dir}/Plots/Tracer_stereo/{timeind:04}.pdf')

    except:
        print(f"didn't like {timeind}")

print(f'Plotted for \n {file}')