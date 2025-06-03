import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
from tomplot import ( tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     tomplot_field_title, extract_gusto_field,
                     extract_gusto_coords)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #n
# When copying this example these should not be relative to this file
results_dir = '/home/d-witt/firedrake/src/gusto/Scorer_param_testing/results/testing_linear_scorer'
plots_dir = f'{results_dir}/plots'

run_head = 'mountain_gw'
plot_stem = f'{plots_dir}/mountain_gw_scorer_param_linear.png'

# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
field_names =  ['theta_perturbation', 'Brunt-Vaisala_squared', 'u_z' ,'ScorerParameter']
titles =  ['theta perturbation', 'Brunt-Vaisala squared','Vertical velocity' ,'Scorer Parameter']
field_label = r'$b \ / $ m s$^{-1}$'
colour_scheme = 'jet'
# Things that are the same for all subplots
contour_method = 'tricontour'

# 1D grids for vertical regridding
coords_lon_1d = np.linspace(-180, 180, 50)
coords_lat_1d = np.linspace(-90, 90, 50)
# Dictionary to hold plotting grids -- keys are "slice_along" values
plotting_grids = {'lat': coords_lon_1d, 'lon': coords_lat_1d}
results_path = f'{results_dir}/field_output.nc'
# Level for horizontal slices
# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #

time_idxs = [ -1]
for time_idx in time_idxs:
    fig, axarray = plt.subplots(1, 4, figsize=(12, 8), sharey='row', sharex=True)
    # Loop through subplots
    for i, (ax, field_name) in enumerate(zip(axarray.flatten(), field_names)):

        data_file = Dataset(results_path, 'r')
        # ------------------------------------------------------------------------ #
        # Data extraction
        # ------------------------------------------------------------------------ #
        field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx) 
        coords_X, coords_Y = extract_gusto_coords(data_file, field_name)

        # extract a cross sectional set of values
        time = data_file['time'][time_idx]
        # ------------------------------------------------------------------------ #
        # Plot data
        # ------------------------------------------------------------------------ #
        contours = tomplot_contours(field_data)
        cmap, lines = tomplot_cmap(contours, colour_scheme)
        cf, _ = plot_contoured_field(ax, coords_X, coords_Y, field_data, contour_method,
                                    contours, cmap=cmap, line_contours=lines)
        add_colorbar_ax(ax, cf, field_label)
        tomplot_field_title(ax, f'plot of {field_name}', minmax=True, field_data=field_data)   
        fig.suptitle(f'lower resolution bouyancy field of the gravity wave at {time} s ')

        ax.set_ylabel('z (km)')
        ax.set_xlabel('x (km)')
    # ---------------------------------------------------------------------------- #
    # Save figure
    # ---------------------------------------------------------------------------- #
    plot_name = f'{plot_stem}_{time}.png'
    print(f'Saving figure to {plot_name}')
    plt.show()
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()

