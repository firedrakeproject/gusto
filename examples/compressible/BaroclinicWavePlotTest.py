"""
A tomplot example, plotting 6 different slices with LFRic data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
from os.path import abspath, dirname
from tomplot import (set_tomplot_style, tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     regrid_vertical_slice, tomplot_field_title,
                     extract_gusto_vertical_slice, apply_gusto_domain,
                     reshape_gusto_data, extract_gusto_field,
                     extract_gusto_coords)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
results_dir = '/home/d-witt/firedrake/src/gusto/results/TomPlotData/Baroclinicpert_extradiag_long'
plot_dir = '/home/d-witt/firedrake/src/gusto/results/TomPlotPlots'
results_file_name = f'{results_dir}/field_output.nc'
plot_stem = f'{plot_dir}/zoomed'

# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
# Specify lists for variables that are different between subplots
field_names = ['u_zonal', 'u_meridional',
               'Temperature', 'Pressure_Vt']
titles = ['Zonal Velocity', 'Meridional Velocity',
          'Temperature (850 HPa)', 'Surface Pressure']
slice_along_values = ['z', 'z',
                      'z', 'z']
field_labels = [ r'$u \ / $ m s$^{-1}$', r'$v \ / $ m s$^{-1}$',
                r'$T \ / $K', r'$P \ / $Pa']
remove_contour_vals = [None, None, None, None]
lon_min = 0
lon_max = 120
lat_min = 0
lat_max = 90
colour_schemes = ['OrRd', 'RdBu_r', 'plasma', 'plasma']

# Things that are the same for all subplots
time_idxs = [0,8,16,24,32,40,48,-1]
contour_method = 'tricontour'
slice_at_val = [0, 0 , 1500, 0]
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
data_file = Dataset(results_file_name, 'r')
for time_idx in time_idxs:
    fig, axarray = plt.subplots(2, 2, figsize=(16, 8), sharey='row')

    # Loop through subplots
    for i, (ax, field_name, field_label, colour_scheme, slice_along, remove_contour, slice_at, title) in \
        enumerate(zip(axarray.flatten(), field_names, field_labels,
                    colour_schemes, slice_along_values, remove_contour_vals, slice_at_val, titles)):
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

    
            # bundle these three variables into a pandas data frame and sort for co-ordinates for 
            


            # Take level for a horizontal slice
            field_data = field_full[:,level]

            data_dict = {'field': field_data, 'X': coords_X_full[:,level], 'Y': coords_Y_full[:,level]}
            # place data in pandas
            df = pd.DataFrame(data_dict)

            df = df[(df["X"]>=lon_min) & (df["X"] <= lon_max ) 
                    & (df["Y"]>=lat_min) & (df["X"] <= lat_max)]
            
            field_data = df['field']
            coords_X_zoomed = df['X']
            coords_Y_zoomed = df['Y']

            coords_hori = coords_X_zoomed
            coords_Z = coords_Y_zoomed

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
        if time_idx == 0:
            if i == 1:
                remove_contour = 0
         
        if i == 2:
            contours = np.arange(220, 320, 10)
        else:
            contours = tomplot_contours(field_data)

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
    fig.suptitle(f'SBR at {time_in_days} days')
    # ---------------------------------------------------------------------------- #
    # Save figure
    # ---------------------------------------------------------------------------- #
    plot_name = f'{plot_stem}_{time_idx:00d}.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()


