import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
import matplotlib.path as mpath
from matplotlib import (cm, colors, gridspec)
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import pdb
import os
import functions as fcs

file = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-300sols_tracer_tophat-80_ref-5'

ref_lev = 5

start_sol = 0

results_dir = f'/data/home/sh1293/results/{file}'

if not os.path.exists(f'{results_dir}/Plots/Stereo_ani/'):
    os.makedirs(f'{results_dir}/Plots/Stereo_ani/')

# timeind = -1

ds = xr.open_dataset(f'{results_dir}/regrid_output.nc')
q = ds.PotentialVorticity
try:
    co2 = ds.CO2cond_flag
except:
    co2 = ds.D_minus_H_rel_flag_less
co2_new = ds.CO2cond_flag_cumulative/10
zonmax = fcs.max_zonal_mean(q)
t = ds.tracer


omega = 2*np.pi/88774
hbart = 17000
val = 2*omega/hbart

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

dt = (0.5)**(ref_lev-4) * 4500.

for timeind in range(int(start_sol*88774/dt), len(q.time)):
# for timeind in [153, 154, 155, 156]:
# for timeind in range(0, 10):
# for timeind in [9, 5918]:
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

    contourplot = q[timeind,:,:].plot.contourf(ax=ax, add_colorbar=True,
                                transform = ccrs.PlateCarree(), cmap='OrRd', levels=22, vmin=0, vmax=1.75*val, extend = 'both')


    # contours = ax.contourf(co2[:,:,-1].lon, co2[:,:,-1].lat, co2[:,:,-1].values, levels=[-0.5, 1.5], colors=None, hatches='xx')
    # contours = co2[timeind,:,:].plot.contour(ax=ax, colors=['blue', 'green', 'yellow'], transform=ccrs.PlateCarree(), add_colorbar=False, levels=[0.1, 0.5, 1])
    contours = co2[timeind,:,:].plot.contour(ax=ax, colors=['green'], transform=ccrs.PlateCarree(), add_colorbar=False, levels=[0.5])
    # contours_new = co2_new[timeind,:,:].plot.contour(ax=ax, colors=['yellow'], transform=ccrs.PlateCarree(), add_colorbar=False, levels=[0.5])
    # contours1 = co2[timeind,:,:].plot.contourf(ax=ax, vmin=0, vmax=1, alpha=0.5, cmap='GnBu', extend='both', add_colorbar=True,
    #                                             transform=ccrs.PlateCarree())

    ### this is the circle at the latitude of maximum zonal mean PV
    # gl1 = ax.gridlines(crs = ccrs.PlateCarree(), linewidth = 1, linestyle = '-', color = 'blue', alpha = 1, draw_labels=False)
    # gl1.ylocator = mticker.FixedLocator([zonmax.max_lat[timeind].values])
    # gl1.xlocator = mticker.FixedLocator([])
    plt.rcParams['hatch.color'] = 'deepskyblue'
    contourplot1 = np.log(t[timeind,:,:]).plot.contourf(ax=ax, transform = ccrs.PlateCarree(), add_colorbar=False, colors=['none'], hatches=['++'], levels=[np.log(15e-3),1], extend = 'neither')


    colorbar = contourplot.colorbar
    colorbar.set_label(r'Potential Vorticity ($\frac{2 \Omega}{H}$)')
    ticks1 = [1.75*val, 1.5*val, 1.25*val, 1*val, 0.75*val, 0.5*val, 0.25*val, 0]
    ticks2 = ['1.75', '1.5', '1.25', '1', '0.75', '0.5', '0.25', '0']
    colorbar.set_ticks(ticks=ticks1, labels=ticks2)


    ax.set_title(f'{ds.time.values[timeind]/88774:.4f}sol')

    plt.savefig(f'{results_dir}/Plots/Stereo_ani/{timeind:04}.pdf')

    # print(f'Plot made: \n {results_dir}/Plots/Stereo_ani/{timeind:04}.pdf')

print(f'Plotted for:\n {file}')