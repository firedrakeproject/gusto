"""
A tomplot example, plotting 6 different slices with LFRic data.
"""

import numpy as np
import matplotlib.pyplot as plt
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
results_dir = f'/data/home/sh1293/firedrake-real-opt/src/gusto/examples/shallow_water/results/annular_vortex_mars_60-70'
plot_dir = f'{results_dir}'
results_file_name = f'{results_dir}/field_output.nc'
plot_name = f'{plot_dir}/testing.png'

# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
# Specify lists for variables that are different between subplots
field_names = ['u_zonal', 'u_meridional', 
               'D', 'PotentialVorticity']
            
field_labels = [r'$u \ / $ m s$^{-1}$', r'$v \ / $ m s$^{-1}$', r'D',
                r'PV']
colour_schemes = ['OrRd', 'RdBu_r', 'RdBu_r',
                  'OrRd']
# Things that are the same for all subplots
time_idx = 0
contour_method = 'tricontour'
u_contour = np.arange(-55, 55, 10)
D_contour = np.arange(15500, 18550, 300)
pv_contour = np.arange(-9e-9, 2e-8, 2e-9)

slice_at = 0.0
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
set_tomplot_style(fontsize=12)
data_file = Dataset(results_file_name, 'r')
fig, axarray = plt.subplots(2, 2, figsize=(16, 8), sharey='row')

# Loop through subplots
for i, (ax, field_name, field_label, colour_scheme) in \
    enumerate(zip(axarray.flatten(), field_names, field_labels,
                  colour_schemes)):
    # ------------------------------------------------------------------------ #
    # Data extraction
    # ------------------------------------------------------------------------ #

    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    # ------------------------------------------------------------------------ #
    # Plot data
    # ------------------------------------------------------------------------ #
    if field_name == 'u_zonal':
        contours = u_contour
    elif field_name =='PotentialVorticity':
        contours = pv_contour
    elif field_name == 'D':
        contours = D_contour
    else: 
        contours = tomplot_contours(field_data)


    cmap, lines = tomplot_cmap(contours, colour_scheme)
    cf, _ = plot_contoured_field(ax, coords_X, coords_Y, field_data,
                                 contour_method, contours, cmap=cmap,
                                 line_contours=lines)
    
    # ax.plot(field_data, coords_Y)
    add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)
    # Don't add ylabels unless left-most subplots
    ylabel = True if i % 3 == 0 else None
    ylabelpad = -30 if i > 2 else -10
    apply_gusto_domain(ax, data_file, ylabel=ylabel,
                       xlabelpad=-15, ylabelpad=ylabelpad)
    tomplot_field_title(ax, None, minmax=True, field_data=field_data)

# These subplots tend to be quite clustered together, so move them apart a bit
fig.subplots_adjust(wspace=0.3, hspace=0.3)

# ---------------------------------------------------------------------------- #
# Save figure
# ---------------------------------------------------------------------------- #
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()



