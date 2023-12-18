"""
Plot for a 1x3 graph of comparing the difference between two runs
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
from tomplot import (tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     regrid_vertical_slice, tomplot_field_title,
                     extract_gusto_vertical_slice, 
                     reshape_gusto_data, extract_gusto_field,
                     extract_gusto_coords)
# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
test_case = 'Held Suarez'
results_dir_1 = '/home/d-witt/firedrake/src/gusto/plotting/MyTomplots/HeldSuarez/HS_test'
results_dir_2 = '/home/d-witt/firedrake/src/gusto/plotting/MyTomplots/HeldSuarez/HS_no_relaxation'
plot_dir = '/home/d-witt/firedrake/src/gusto/plotting/MyTomplots/HeldSuarez/ComparisonPlots'
results_file_name_1 = f'{results_dir_1}/field_output.nc'
results_file_name_2 = f'{results_dir_2}/field_output.nc'
plot_stem = f'{plot_dir}/test_vs_unrelaxred'

data_file1 = Dataset(results_file_name_1, 'r')
data_file2 = Dataset(results_file_name_2, 'r')
# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
# Things that iterate through rows
field_names = ['u_zonal', 'theta']
titles = ['zonal velocity', 'theta']
slice_along_values = ['lon', 'lon']
field_labels = [r'$v \ / $ m s$^{-1}$', 'hPa']
remove_contour_vals = [None, None]
colour_schemes = ['jet', 'jet']

# Things that iterate through coloumns
column_titles = ['relaxed', 'unrelaxed']



# Things that are the same for all subplots
time_idxs = np.arange(0,128, 8)
time_idxs = [0, 64, 80, 120]
contour_method = 'tricontour'
slice_at_vals = [0, 0]
# 1D grids for vertical regridding
coords_lon_1d = np.linspace(-180, 180, 50)
coords_lat_1d = np.linspace(-90, 90, 50)
# Dictionary to hold plotting grids -- keys are "slice_along" values
plotting_grids = {'lat': coords_lon_1d, 'lon': coords_lat_1d}
# Level for horizontal slices
level = 0
# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #
#set_tomplot_style(fontsize=12)

compare_data = [] # this will be calculated as we go
data_files = [data_file1, data_file2]
# breakpoint()
n = len(field_names) # how many rows our subplot needs
for time_idx in time_idxs:
    fig, axs = plt.subplots(n, 3, figsize=(16, 8), sharey='row')
    # Loop through rows
    for j, (field_name, field_label, slice_along, remove_contour, slice_at, colour_scheme) in  \
            enumerate(zip(field_names, field_labels,slice_along_values, remove_contour_vals, slice_at_vals, colour_schemes)):
        # Loop through columns
        for i, (title) in enumerate(zip(column_titles)):
            if n == 1:
                ax = axs[i]
            else:
                ax = axs[j,i]
            # ------------------------------------------------------------------------ #
            # Data extraction
            # ------------------------------------------------------------------------ #
            if not i==2: # data extraction only needs to be done for the comparison fields
                data_file = data_files[i]
                if slice_along == 'z':
                    field_full = extract_gusto_field(data_file, field_name, time_idx)
                    coords_X_full, coords_Y_full, coords_Z_full = \
                        extract_gusto_coords(data_file, field_name)

                    # Reshape
                    field_full, coords_X_full, coords_Y_full, _ = \
                        reshape_gusto_data(field_full, coords_X_full,
                                            coords_Y_full, coords_Z_full)

                    # Take level for a horizontal slice
                    field_data = field_full[:,level]
                    coords_hori = coords_X_full[:,level]
                    coords_Z = coords_Y_full[:,level]

                else:
                    orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
                        extract_gusto_vertical_slice(data_file, field_name, time_idx,
                                                    slice_along=slice_along, slice_at=slice_at)

                    # Slices need regridding as points don't cleanly live along lon or lat = 0.0
                    field_data, coords_hori, coords_Z = regrid_vertical_slice(plotting_grids[slice_along],
                                                                            slice_along, slice_at,
                                                                            orig_coords_X, orig_coords_Y,
                                                                            orig_coords_Z, orig_field_data)
                if i == 0:
                    base_data = field_data.copy()
                elif i == 1:
                    compare_data = field_data.copy()
                    compare_coords_hori = coords_hori.copy()
                    compare_coords_z = coords_Z.copy()
            else:

                field_data = base_data - compare_data
                colour_scheme = 'bwr'   
                coords_hori = compare_coords_hori
                coords_Z = compare_coords_z
            
            time = data_file['time'][time_idx]
            time_in_days = time / (24*60*60)
            # ------------------------------------------------------------------------ #
            # Plot data
            # ------------------------------------------------------------------------ #
            
            contours = tomplot_contours(field_data)
            if field_name=='u_zonal' and time_idx==0:
                contours=np.arange(0,2,1)
            cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=remove_contour)
            cf, _ = plot_contoured_field(ax, coords_hori, coords_Z, field_data,
                                        contour_method, contours, cmap=cmap,
                                        line_contours=lines)
            add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)
            # Don't add ylabels unless left-most subplots
            ylabel = True if i % 3 == 0 else None
            ylabelpad = -30 if i > 2 else -10

            tomplot_field_title(ax, title, minmax=True, field_data=field_data)

    # These subplots tend to be quite clustered together, so move them apart a bit
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(f'{test_case} at {time_in_days} days')
    # ---------------------------------------------------------------------------- #
    # Save figure
    # ---------------------------------------------------------------------------- #
    plot_name = f'{plot_stem}_{time_in_days}_days.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()