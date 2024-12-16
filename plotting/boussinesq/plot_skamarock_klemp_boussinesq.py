"""
Plots the Boussinesq Skamarock-Klemp gravity wave in a vertical slice. This can
plot the compressible, incompressible and linear cases.

This plots the initial conditions @ t = 0 s, with
(a) buoyancy perturbation, (b) buoyancy
and the final state @ t = 3600 s, with
(a) buoyancy perturbation,
(b) a 1D slice through the wave
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, tomplot_field_title, extract_gusto_coords,
    extract_gusto_field, reshape_gusto_data, add_colorbar_fig
)

# Can be incompressible/compressible/linear
test = 'skamarock_klemp_linear_bouss'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/boussinesq/{test}'

# ---------------------------------------------------------------------------- #
# Initial plot details
# ---------------------------------------------------------------------------- #
init_field_names = ['b_perturbation', 'b']
init_colour_schemes = ['YlOrRd', 'Purples']
init_field_labels = [r'$\Delta b$ (m s$^{-2}$)', r'$b$ (m s$^{-2}$)']
init_contours = [np.linspace(0.0, 0.01, 11), np.linspace(0, 1, 11)]
init_contours_to_remove = [None, None]

# ---------------------------------------------------------------------------- #
# Final plot details
# ---------------------------------------------------------------------------- #
final_field_name = 'b_perturbation'
final_colour_scheme = 'RdBu_r'
final_field_label = r'$\Delta b$ (m s$^{-2}$)'
final_contours = np.linspace(-3.0e-3, 3.0e-3, 13)
final_contour_to_remove = 0.0

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'tricontour'
xlims = [0, 300.0]
ylims = [0, 10.0]

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
fig.subplots_adjust(wspace=0.2)
plot_name = f'{plot_stem}_initial.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------- #
# FINAL PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(2, 1, figsize=(8, 8), sharex='all')
time_idx = -1

# Data extraction ----------------------------------------------------------
field_data = extract_gusto_field(data_file, final_field_name, time_idx=time_idx)
coords_X, coords_Y = extract_gusto_coords(data_file, final_field_name)
time = data_file['time'][time_idx]

# Plot 2D data -----------------------------------------------------------------
ax = axarray[0]

cmap, lines = tomplot_cmap(
    final_contours, final_colour_scheme, remove_contour=final_contour_to_remove
)
cf, lines = plot_contoured_field(
    ax, coords_X, coords_Y, field_data, contour_method, final_contours,
    cmap=cmap, line_contours=lines
)

add_colorbar_fig(
    fig, cf, final_field_label, ax_idxs=[0], location='right', cbar_labelpad=-40
)
tomplot_field_title(
    ax, f't = {time:.1f} s', minmax=True, field_data=field_data
)

ax.set_ylabel(r'$z$ (km)', labelpad=-20)
ax.set_ylim(ylims)
ax.set_yticks(ylims)
ax.set_yticklabels(ylims)

# Plot 1D data -----------------------------------------------------------------
ax = axarray[1]

field_data, coords_X, coords_Y = reshape_gusto_data(field_data, coords_X, coords_Y)

# Determine midpoint index
mid_idx = np.floor_divide(np.shape(field_data)[1], 2)
slice_height = coords_Y[0, mid_idx]

ax.plot(coords_X[:, mid_idx], field_data[:, mid_idx], color='black')

tomplot_field_title(
    ax, r'$z$' + f' = {slice_height} km'
)

b_lims = [np.min(final_contours), np.max(final_contours)]

ax.set_ylabel(final_field_label, labelpad=-20)
ax.set_ylim(b_lims)
ax.set_yticks(b_lims)
ax.set_yticklabels(b_lims)

ax.set_xlabel(r'$x$ (km)', labelpad=-10)
ax.set_xlim(xlims)
ax.set_xticks(xlims)
ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
fig.subplots_adjust(hspace=0.2)
plot_name = f'{plot_stem}_final.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
