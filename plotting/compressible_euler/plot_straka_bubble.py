"""
Plots the Straka bubble test case.

This plots:
(a) theta perturbation @ t = 0 s, (b) theta perturbation @ t = 900 s
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_fig, tomplot_field_title, extract_gusto_coords,
    extract_gusto_field
)

test = 'straka_bubble'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/compressible_euler/{test}'

# ---------------------------------------------------------------------------- #
# Plot details
# ---------------------------------------------------------------------------- #
field_names = ['theta_perturbation', 'theta_perturbation']
time_idxs = [0, -1]
cbars = [False, True]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contours = np.linspace(-7.5, 0.5, 17)
colour_scheme = 'Blues_r'
field_label = r'$\Delta \theta$ (K)'
contour_method = 'tricontour'
xlims = [25.6, 38.4]
ylims = [0., 5.0]

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(1, 2, figsize=(18, 6), sharex='all', sharey='all')
time_idx = 0

for i, (ax, time_idx, field_name, cbar) in \
        enumerate(zip(axarray.flatten(), time_idxs, field_names, cbars)):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx]

    # Plot data ----------------------------------------------------------------
    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=0.0)
    cf, lines = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines, negative_linestyles='solid'
    )

    if cbar:
        add_colorbar_fig(
            fig, cf, field_label, ax_idxs=[i], location='right'
        )
    tomplot_field_title(
        ax, f't = {time:.1f} s', minmax=True, field_data=field_data
    )

    # Labels -------------------------------------------------------------------
    if i == 0:
        ax.set_ylabel(r'$z$ (km)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    ax.set_xlabel(r'$x$ (km)', labelpad=-10)
    ax.set_xlim(xlims)
    ax.set_xticks(xlims)
    ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
fig.subplots_adjust(wspace=0.15)
plot_name = f'{plot_stem}.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
