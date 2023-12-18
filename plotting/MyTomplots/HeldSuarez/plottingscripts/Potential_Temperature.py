"""
A tomplot example, for making a quick single plot from Gusto data.
"""

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pandas as pd
import numpy as np
from tomplot import (tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     tomplot_field_title, extract_gusto_coords,
                     extract_gusto_field, apply_gusto_domain, reshape_gusto_data,
                     extract_gusto_vertical_slice, regrid_vertical_slice)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
results_dir = '/home/d-witt/firedrake/src/gusto/plotting/MyTomplots/HeldSuarez/HS_test/Held_suarez_test'
plot_dir = '/home/d-witt/firedrake/src/gusto/plotting/MyTomplots/HeldSuarez/HS_test/Held_suarez_test'
results_file_name = f'{results_dir}/field_output.nc'
plot_stem = f'{plot_dir}/potentialTemperature_uzonal'

# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
field_names = ['Temperature', 'theta']
colour_schemes = ['RdBu_r', 'RdBu_r']
time_idxs = np.arange(0,128,4)
time_idxs = [0, -1]
field_labels = [' potential temp', 'temp']
contour_method = 'tricontour'
contour_to_remove=None
slice_along_values = ['lon', 'lon']
level = 0 # level for horizontal slice
slice_at_values = [0, 0]
# 1D grids for vertical regridding
coords_lon_1d = np.linspace(-180, 180, 50)
coords_lat_1d = np.linspace(-90, 90, 50)
# Dictionary to hold plotting grids -- keys are "slice_along" values
plotting_grids = {'lat': coords_lon_1d, 'lon': coords_lat_1d}
breakpoint()

# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #

data_file = Dataset(results_file_name, 'r')

for time_idx in time_idxs:
    fig, axarray = plt.subplots(1, 2, figsize=(16, 8))
    plot_name=f'{plot_stem}'
    # Loop through subplots
    for i, (ax, field_name, field_label, colour_scheme, slice_along, slice_at, title) in \
        enumerate(zip(axarray.flatten(), field_names, field_labels,
                    colour_schemes, slice_along_values, slice_at_values, field_labels)):

    # ---------------------------------------------------------------------------- #
        # Data extraction
        if slice_along == 'z':
                field_full = extract_gusto_field(data_file, field_name, time_idx)
                coords_X_full, coords_Y_full, coords_Z_full = \
                    extract_gusto_coords(data_file, field_name)

                # Reshape
                field_full, coords_X_full, coords_Y_full, _ = \
                    reshape_gusto_data(field_full, coords_X_full,
                                        coords_Y_full, coords_Z_full)

                field_data = field_full[:,level]
                coords_X = coords_X_full[:level]
                coords_Y = coords_Y_full[:level]

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
        
        # ---------------------------------------------------------------------------- #
        # Plot data
        # ---------------------------------------------------------------------------- #
     
        contours = tomplot_contours(field_data)
        if field_name == 'u_zonal' and time_idx == 0: 
             contours = np.arange(0,2,1)
        cmap, lines = tomplot_cmap(contours, colour_scheme)
        cf, _ = plot_contoured_field(ax, coords_hori.flatten(), coords_Z.flatten(), field_data.flatten(),
                                    contour_method, contours, cmap=cmap,
                                    line_contours=lines)
        add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)
        # Don't add ylabels unless left-most subplots
        ylabel = True if i % 3 == 0 else None
        ylabelpad = -30 if i > 2 else -10

        tomplot_field_title(ax, None, minmax=True, field_data=field_data)
    # ---------------------------------------------------------------------------- #
    # Save figure
    # ---------------------------------------------------------------------------- #
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(f'Potential Temperature and Temperature at {time_in_days} days')

    plot_name = f'{plot_name}_at_{time_in_days}days.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()
