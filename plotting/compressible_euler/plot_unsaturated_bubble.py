"""
Plots the unsaturated moist rising bubble test case, which features rain.

This plots the initial conditions @ t = 0 s, with
(a) theta perturbation, (b) relative humidity,
and the final state @ t = 600 s, with
(a) theta perturbation, (b) relative humidity and (c) rain mixing ratio
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, tomplot_field_title, extract_gusto_coords,
    extract_gusto_field
)

test = 'unsaturated_bubble'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/compressible_euler/{test}'

# ---------------------------------------------------------------------------- #
# Initial plot details
# ---------------------------------------------------------------------------- #
init_field_names = ['theta_perturbation', 'RelativeHumidity']
init_colour_schemes = ['Reds', 'Blues']
init_field_labels = [r'$\Delta\theta$ (K)', 'Relative Humidity']
init_contours = [np.linspace(-0.25, 3.0, 14), np.linspace(0.0, 1.1, 12)]
init_contours_to_remove = [0.0, 0.2]

# ---------------------------------------------------------------------------- #
# Final plot details
# ---------------------------------------------------------------------------- #
final_field_names = ['theta_perturbation', 'RelativeHumidity', 'rain']
final_colour_schemes = ['RdBu_r', 'Blues', 'Purples']
final_field_labels = [r'$\Delta\theta$ (K)', 'Relative Humidity', r'$m_r$ (kg/kg)']
final_contours = [np.linspace(-3.5, 3.5, 15),
                  np.linspace(0.0, 1.1, 12),
                  np.linspace(-2.5e-6, 5.0e-5, 12)]
final_contours_to_remove = [0.0, None, None]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'tricontour'
xlims = [0, 3.6]
ylims = [0, 2.4]

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# INITIAL PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(1, 2, figsize=(12, 6), sharex='all', sharey='all')
time_idx = 0

for i, (ax, field_name, field_label, colour_scheme, contours, to_remove) in \
        enumerate(zip(axarray.flatten(), init_field_names, init_field_labels,
                      init_colour_schemes, init_contours, init_contours_to_remove)):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx]

    # Plot data ----------------------------------------------------------------
    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=to_remove)
    cf, lines = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(
        fig, cf, field_label, location='bottom', cbar_labelpad=-10
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
plot_name = f'{plot_stem}_initial.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------- #
# FINAL PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(1, 3, figsize=(18, 6), sharex='all', sharey='all')
time_idx = -1

for i, (ax, field_name, field_label, colour_scheme, contours, to_remove) in \
        enumerate(zip(axarray.flatten(), final_field_names, final_field_labels,
                      final_colour_schemes, final_contours,
                      final_contours_to_remove)):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx]

    # Plot data ----------------------------------------------------------------
    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=to_remove)
    cf, lines = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(
        fig, cf, field_label, location='bottom', cbar_labelpad=-10
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
fig.subplots_adjust(wspace=0.18)
plot_name = f'{plot_stem}_final.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
