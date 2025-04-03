"""
Plots the fields from the 1D shallow water wave.

This plots:
(a) u @ t = 0 s, (b) v @ t = 0 s, (c) D @ t = 0 s
(d) u @ t = 1 s, (e) v @ t = 1 s, (f) D @ t = 1 s
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from tomplot import (
    set_tomplot_style, tomplot_field_title, extract_gusto_coords,
    extract_gusto_field
)

test = 'shallow_water_1d_wave'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/shallow_water/{test}'

# ---------------------------------------------------------------------------- #
# Plot details
# ---------------------------------------------------------------------------- #
field_names = ['u', 'v', 'D', 'u', 'v', 'D']
time_idxs = [0, 0, 0, -1, -1, -1]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
xlims = [0, 2*np.pi]
xlims_labels = [0, r'$2\pi$']

ylims = {
    'u': [-0.5, 0.5],
    'v': [-0.5, 0.5],
    'D': [8, 12]
}
field_labels = {
    'u': r'$u$ (m s$^{-1}$)',
    'v': r'$v$ (m s$^{-1}$)',
    'D': r'$D$ (m)'
}

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(2, 3, figsize=(16, 6), sharex='all', sharey='col')

for i, (ax, time_idx, field_name) in \
        enumerate(zip(axarray.flatten(), time_idxs, field_names)):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx]

    # Get coordinates in order
    data_frame = pd.DataFrame({'coords': coords_X, 'field': field_data})
    data_frame = data_frame.sort_values(by=['coords'])
    coords_X = data_frame['coords']
    field_data = data_frame['field']

    # Convert coordinates to m
    coords_X *= 1000.

    # Plot data ----------------------------------------------------------------
    ax.plot(coords_X, field_data, color='black', linestyle='-', marker='')

    tomplot_field_title(
        ax, f't = {time:.1f} s', minmax=True, field_data=field_data,
        minmax_format='1.2f'
    )

    # Labels -------------------------------------------------------------------
    ax.set_ylabel(field_labels[field_name], labelpad=-15)
    ax.set_ylim(ylims[field_name])
    ax.set_yticks(ylims[field_name])
    ax.set_yticklabels(ylims[field_name])

    if i > 2:
        ax.set_xlabel(r'$x$ (m)', labelpad=-10)
        ax.set_xlim(xlims)
        ax.set_xticks(xlims)
        ax.set_xticklabels(xlims_labels)

# Save figure ------------------------------------------------------------------
fig.subplots_adjust(wspace=0.2)
plot_name = f'{plot_stem}.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
