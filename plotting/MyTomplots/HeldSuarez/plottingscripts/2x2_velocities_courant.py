"""
Plot for a 2x2 grpah of zonal velocity, temperature, theta, pressure 
"""
from venv import create
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tomplot import (area_restriction, tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     regrid_vertical_slice, tomplot_field_title,
                     extract_gusto_vertical_slice, 
                     reshape_gusto_data, extract_gusto_field,
                     extract_gusto_coords, create_animation, check_directory)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #n
# When copying this example these should not be relative to this file
results_dir = '/home/d-witt/firedrake/src/gusto/plotting/MyTomplots/HeldSuarez/HS_C4_sigmainterpolation'
plot_dir = f'{results_dir}/courant_vel'
check_directory(plot_dir)
results_file_name = f'{results_dir}/field_output.nc'
plot_stem = f'{plot_dir}/vel_courant'
# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
# Specify lists for variables that are different between subplots
field_names = ['u_zonal', 'u_meridional', 
               'u_radial', 'CourantNumber']
titles = ['u_zonal', 'u_meridional', 
          'u_radial', 'Courant Number']
slice_along_values = ['lon', 'lon',
                      'lon', 'lon']
field_labels = [ r'$u \ / $ m s$^{-1}$', r'$v \ / $ m s$^{-1}$',
                r'$r \ / $m s${-1}$', r'']
remove_contour_vals = [None, None, None, None]
domain_limit = {'X':(0, 180), 'Y':(0,90)}
colour_schemes = ['jet', 'jet', 'jet', 'jet']
# Things that are the same for all subplots
time_idxs = np.arange(0, 17*160+1, 160) # plots every twenty days

contour_method = 'tricontour'
slice_at_vals = [0, 0 , 0, 0]
# 1D grids for vertical regridding
coords_lon_1d = np.linspace(-180, 180, 50)
coords_lat_1d = np.linspace(-90, 90, 50)
# Dictionary to hold plotting grids -- keys are "slice_along" values
plotting_grids = {'lat': coords_lon_1d, 'lon': coords_lat_1d}
levels = [0, 0, 0, 0]
# Level for horizontal slices
# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #

data_file = Dataset(results_file_name, 'r')
time = data_file.variables['time'][:]
time_idxs = np.arange(0, len(data_file.variables['time'][:]), 8)


for time_idx in time_idxs:
    fig, axarray = plt.subplots(2, 2, figsize=(16, 8), sharey='row')

    # Loop through subplots
    for i, (ax, field_name, field_label, colour_scheme, slice_along, remove_contour, slice_at, title, level) in \
        enumerate(zip(axarray.flatten(), field_names, field_labels,
                    colour_schemes, slice_along_values, remove_contour_vals, slice_at_vals, titles, levels)):
        # ------------------------------------------------------------------------ #
        # Data extraction
        # ------------------------------------------------------------------------ #
        if slice_along == 'z':
            field_full = extract_gusto_field(data_file, field_name, time_idx)
            coords_X_full, coords_Y_full, coords_Z_full = \
                extract_gusto_coords(data_file, field_name)

            # Reshape
            field_full, coords_X_full, coords_Y_full, _ = \
                reshape_gusto_data(field_full, coords_X_full,
                                    coords_Y_full, coords_Z_full)

            field_data, coords_hori, coords_Z = \
                area_restriction(field_full[:,level], coords_X_full[:,level], 
                                 coords_Y_full[:,level], domain_limit)
        
        else:
            orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
                extract_gusto_vertical_slice(data_file, field_name, time_idx,
                                            slice_along=slice_along, slice_at=slice_at)

            # Slices need regridding as points don't cleanly live along lon or lat = 0.0
            field_data, coords_hori, coords_Z = regrid_vertical_slice(plotting_grids[slice_along],
                                                                    slice_along, slice_at,
                                                                    orig_coords_X, orig_coords_Y,
                                                                    orig_coords_Z, orig_field_data)
        time = data_file['time'][time_idx]
        time_in_days = time / (24*60*60)
        # ------------------------------------------------------------------------ #
        # Plot data
        # ------------------------------------------------------------------------ #
        if field_name =="CourantNumber":
            contour = np.arange(0, 0.3, 0.02)
        elif field_name == 'u_zonal':
            contour = np.arange(-15, 35, 5)
        elif field_name == 'u_radial':
            contour = np.arange(-1, 1, 0.2)
        else:
            contour = tomplot_contours(field_data)

        cmap, lines = tomplot_cmap(contour, colour_scheme, remove_contour=remove_contour)
        cf, _ = plot_contoured_field(ax, coords_hori, coords_Z, field_data,
                                    contour_method, contour, cmap=cmap,
                                    line_contours=lines)
        add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)
        # Don't add ylabels unless left-most subplots
        ylabel = True if i % 3 == 0 else None
        ylabelpad = -30 if i > 2 else -10

        tomplot_field_title(ax, title, minmax=True, field_data=field_data)

    # These subplots tend to be quite clustered together, so move them apart a bit
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(f'Baroclinic Wave at {time_in_days} days')
    # ---------------------------------------------------------------------------- #
    # Save figure
    # ---------------------------------------------------------------------------- #
    plot_name = f'{plot_stem}_{time_idx}.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()

create_animation(results_dir, plot_dir, file_name='courant_and_velocity.gif', delay=30)
