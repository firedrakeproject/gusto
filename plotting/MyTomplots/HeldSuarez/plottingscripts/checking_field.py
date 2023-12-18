"""
A tomplot example, for making a quick single plot from Gusto data.
"""

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from tomplot import (tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     tomplot_field_title, extract_gusto_coords,
                     extract_gusto_field, reshape_gusto_data,
                     extract_gusto_vertical_slice, regrid_vertical_slice)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
results_dir = '/home/d-witt/firedrake/src/gusto/results/relaxation_printing2/'
plot_dir = '/home/d-witt/firedrake/src/gusto/results/relaxation_printing2/'
results_file_name = f'{results_dir}/field_output.nc'
plot_stem = f'{plot_dir}/Relaxation'
data_file = Dataset(results_file_name, 'r')
breakpoint()
# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
field_name = 'Sigma'
colour_scheme = 'RdBu_r'
time_idx = 0
field_label = 'temp'
contour_method = 'tricontour'
contour_to_remove=None
slice_along = 'lon'
slice_at = 0
contour = np.linspace(210, 310, 10)
# 1D grids for vertical regridding
coords_lon_1d = np.linspace(-180, 180, 50)
coords_lat_1d = np.linspace(-90, 90, 50)
# Dictionary to hold plotting grids -- keys are "slice_along" values
plotting_grids = {'lat': coords_lon_1d, 'lon': coords_lat_1d}

# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #

data_file = Dataset(results_file_name, 'r')

plot_name=f'{plot_stem}_{field_name}'
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

        # bundle these three variables into a pandas data frame and sort for co-ordinates for 
        # Take level for a horizontal slice
        field_data = field_full[:,level]

        data_dict = {'field': field_data, 'X': coords_X_full[:,level], 'Y': coords_Y_full[:,level]}
        # place data in pandas


else:
    orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
        extract_gusto_vertical_slice(data_file, field_name, time_idx,
                                    slice_along=slice_along, slice_at=slice_at)

    # Slices need regridding as points don't cleanly live along lon or lat = 0.0
    field_data, coords_X, coords_Y = regrid_vertical_slice(plotting_grids[slice_along],
                                                            slice_along, slice_at,
                                                            orig_coords_X, orig_coords_Y,
                                                            orig_coords_Z, orig_field_data)
time = data_file['time'][time_idx]
time_in_days = time / (24*60*60)
# ---------------------------------------------------------------------------- #
# Plot data
# ---------------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
contours = tomplot_contours(field_data)
cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=contour_to_remove)
cf, _ = plot_contoured_field(ax, coords_X, coords_Y, field_data, contour_method,
                            contours, cmap=cmap, line_contours=lines)
add_colorbar_ax(ax, cf, field_label)
tomplot_field_title(ax, f't = {time:.1f}', minmax=True, field_data=field_data)
# ---------------------------------------------------------------------------- #
# Save figure
# ---------------------------------------------------------------------------- #
plot_name = f'{plot_name}_at {level}m.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
