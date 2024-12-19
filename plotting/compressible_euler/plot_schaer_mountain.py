"""
Plots the Schär mountain test case.

This plots:
(a) w @ t = 5 hr, (b) theta perturbation @ t = 5 hr
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, tomplot_field_title, tomplot_contours,
    extract_gusto_coords, extract_gusto_field, reshape_gusto_data
)

test = 'schaer_alpha_0p51'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/compressible_euler/{test}'

# ---------------------------------------------------------------------------- #
# Final plot details
# ---------------------------------------------------------------------------- #
final_field_names = ['u_z', 'theta_perturbation', 'u_z', 'theta_perturbation']
final_colour_schemes = ['PiYG', 'RdBu_r', 'PiYG', 'RdBu_r']
final_field_labels = [
    r'$w$ (m s$^{-1}$)', r'$\Delta\theta$ (K)',
    r'$w$ (m s$^{-1}$)', r'$\Delta\theta$ (K)'
]
final_contours = [
    np.linspace(-1.0, 1.0, 21), np.linspace(-1.0, 1.0, 21),
    np.linspace(-1.0, 1.0, 21), np.linspace(-1.0, 1.0, 21)
]

# ---------------------------------------------------------------------------- #
# Initial plot details
# ---------------------------------------------------------------------------- #
initial_field_names = ['Exner', 'theta']
initial_colour_schemes = ['PuBu', 'Reds']
initial_field_labels = [r'$\Pi$', r'$\theta$ (K)']

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'contour'  # Need to use this method to show mountains!
xlims = [0., 100.]
ylims = [0., 30.]

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# INITIAL PLOTTING
# ---------------------------------------------------------------------------- #

fig, axarray = plt.subplots(1, 2, figsize=(18, 6), sharex='all', sharey='all')
time_idx = 0

for i, (ax, field_name, colour_scheme, field_label) in \
        enumerate(zip(
            axarray.flatten(), initial_field_names, initial_colour_schemes,
            initial_field_labels
        )):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    field_data, coords_X, coords_Y = \
        reshape_gusto_data(field_data, coords_X, coords_Y)
    time = data_file['time'][time_idx]

    contours = tomplot_contours(field_data)
    cmap, lines = tomplot_cmap(contours, colour_scheme)

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(ax, cf, field_label, location='bottom')
    tomplot_field_title(ax, None, minmax=True, field_data=field_data)

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
fig.suptitle(f't = {time:.1f} s')
fig.subplots_adjust(wspace=0.25)
plot_name = f'{plot_stem}_initial.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------- #
# FINAL PLOTTING
# ---------------------------------------------------------------------------- #
xlims_zoom = [30., 70.]
ylims_zoom = [0., 12.]

fig, axarray = plt.subplots(2, 2, figsize=(18, 12), sharex='row', sharey='row')
time_idx = 1

for i, (ax, field_name, colour_scheme, field_label, contours) in \
        enumerate(zip(
            axarray.flatten(), final_field_names, final_colour_schemes,
            final_field_labels, final_contours
        )):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx]

    # # Filter data for panels that are zoomed in mountain region
    # if i in [2, 3]:
    #     data_dict = {
    #         'X': coords_X,
    #         'Y': coords_Y,
    #         'field': field_data
    #     }
    #     data_frame = pd.DataFrame(data_dict)

    #     data_frame = data_frame[
    #         (data_frame['X'] >= xlims_zoom[0])
    #         & (data_frame['X'] <= xlims_zoom[1])
    #         & (data_frame['Y'] >= ylims_zoom[0])
    #         & (data_frame['Y'] <= ylims_zoom[1])
    #     ]
    #     field_data = data_frame['field'].values[:]
    #     coords_X = data_frame['X'].values[:]
    #     coords_Y = data_frame['Y'].values[:]

    field_data, coords_X, coords_Y = \
        reshape_gusto_data(field_data, coords_X, coords_Y)

    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=0.0)

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(ax, cf, field_label, location='bottom')
    tomplot_field_title(
        ax, None, minmax=True, field_data=field_data, minmax_format='.3f'
    )

    # Labels -------------------------------------------------------------------
    ax.set_xlabel(r'$x$ (km)', labelpad=-10)

    if i in [0, 1]:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xticks(xlims)
        ax.set_xticklabels(xlims)
    else:
        ax.set_xlim(xlims_zoom)
        ax.set_ylim(ylims_zoom)
        ax.set_xticks(xlims_zoom)
        ax.set_xticklabels(xlims_zoom)

    if i == 0:
        ax.set_ylabel(r'$z$ (km)', labelpad=-20)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    elif i == 2:
        ax.set_ylabel(r'$z$ (km)', labelpad=-20)
        ax.set_yticks(ylims_zoom)
        ax.set_yticklabels(ylims_zoom)


# Save figure ------------------------------------------------------------------
fig.suptitle(f't = {time:.1f} s')
fig.subplots_adjust(wspace=0.25)
plot_name = f'{plot_stem}_final.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
