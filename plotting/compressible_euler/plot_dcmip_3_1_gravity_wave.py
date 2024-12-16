"""
Plots the DCMIP 3-1 gravity wave test case.

This plots the initial conditions @ t = 0 s, with
(a) zonal wind, (b) theta (c) theta perturbation: all on a lon-lat slice,
(d) zonal wind, (e) theta (f) theta perturbation: on a lat-z slice,

and the final state @ t = 3600 s, with
(a) theta perturbation on a lon-lat slice,
(b) theta perturbation on a lon-z slice.
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, tomplot_field_title, extract_gusto_coords,
    extract_gusto_field, reshape_gusto_data, extract_gusto_vertical_slice,
    regrid_vertical_slice
)

test = 'dcmip_3_1_gravity_wave'

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
init_field_names = ['u_zonal', 'theta', 'theta_perturbation',
                    'u_zonal', 'theta', 'theta_perturbation']
init_colour_schemes = ['YlOrBr', 'PuRd', 'OrRd',
                       'YlOrBr', 'PuRd', 'OrRd',]
init_field_labels = [r'$u$ (m s$^{-1}$)', r'$\theta$ (K)', r'$\Delta\theta$ (K)',
                     r'$u$ (m s$^{-1}$)', r'$\theta$ (K)', r'$\Delta\theta$ (K)']
init_contours = [np.linspace(0, 25, 11),
                 np.linspace(300, 335, 13),
                 np.linspace(0.0, 1.0, 11),
                 np.linspace(0, 25, 11),
                 np.linspace(300, 335, 13),
                 np.linspace(0.0, 1.0, 11)]
init_contours_to_remove = [None, None, None, None, None, None]
init_slice_along = ['z', 'z', 'z', 'lon', 'lon', 'lon']

# ---------------------------------------------------------------------------- #
# Final plot details
# ---------------------------------------------------------------------------- #
final_field_names = ['theta_perturbation', 'theta_perturbation']
final_colour_schemes = ['RdBu_r', 'RdBu_r']
final_field_labels = [r'$\Delta\theta$ (K)', r'$\Delta\theta$ (K)']
final_contours = [np.linspace(-0.1, 0.1, 21),
                  np.linspace(-0.1, 0.1, 21)]
final_contours_to_remove = [0.0, 0.0]
final_slice_along = ['z', 'lat']

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'tricontour'
lon_lims = [-180, 180]
lat_lims = [-90, 90]
z_lims = [0, 10]
level = 5
slice_at_lon = 120.0
slice_at_lat = 0.0

# 1D grids for vertical regridding
coords_lon_1d = np.linspace(-180, 180, 50)
coords_lat_1d = np.linspace(-90, 90, 50)
# Dictionary to hold plotting grids -- keys are "slice_along" values
plotting_grid = {'lat': coords_lon_1d, 'lon': coords_lat_1d}

cbar_format = {'u_zonal': '1.0f',
               'theta': '1.0f',
               'theta_perturbation': '1.1f'}

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# INITIAL PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(2, 3, figsize=(18, 12), sharex='row', sharey='row')
time_idx = 0

for i, (ax, field_name, field_label, colour_scheme, contours,
        to_remove, slice_along) in \
        enumerate(zip(axarray.flatten(), init_field_names, init_field_labels,
                      init_colour_schemes, init_contours,
                      init_contours_to_remove, init_slice_along)):

    # Data extraction ----------------------------------------------------------
    time = data_file['time'][time_idx]

    if slice_along == 'z':
        field_full = extract_gusto_field(data_file, field_name, time_idx)
        coords_X_full, coords_Y_full, coords_Z_full = \
            extract_gusto_coords(data_file, field_name)

        # Reshape
        field_full, coords_X_full, coords_Y_full, _ = \
            reshape_gusto_data(field_full, coords_X_full,
                               coords_Y_full, coords_Z_full)

        # Take level for a horizontal slice
        field_data = field_full[:, level]
        # Abuse of names for coord variables but simplifies code below
        coords_X = coords_X_full[:, level]
        coords_Y = coords_Y_full[:, level]

    else:
        orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
            extract_gusto_vertical_slice(
                data_file, field_name, time_idx,
                slice_along=slice_along, slice_at=slice_at_lon
            )

        # Slices need regridding as points don't cleanly live along lon or lat = 0.0
        field_data, coords_X, coords_Y = \
            regrid_vertical_slice(
                plotting_grid[slice_along], slice_along, slice_at_lon,
                orig_coords_X, orig_coords_Y, orig_coords_Z, orig_field_data
            )
        # Scale coordinates
        coords_Y /= 1000.

    # Plot data ----------------------------------------------------------------
    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=to_remove)
    cf, lines = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(
        fig, cf, field_label, location='bottom', cbar_labelpad=-10,
        cbar_format=cbar_format[field_name]
    )
    if slice_along == 'z':
        tomplot_field_title(
            ax, '$z = $ 5 km', minmax=True, field_data=field_data
        )
    elif slice_along == 'lon':
        tomplot_field_title(
            ax, r'$\lambda = $ 120 deg', minmax=True, field_data=field_data
        )

    # Labels -------------------------------------------------------------------
    if i == 0:
        ax.set_ylabel(r'$\vartheta$ (deg)', labelpad=-20)
        ax.set_ylim(lat_lims)
        ax.set_yticks(lat_lims)
        ax.set_yticklabels(lat_lims)
    elif i == 3:
        ax.set_ylabel(r'$z$ (km)', labelpad=-20)
        ax.set_ylim(z_lims)
        ax.set_yticks(z_lims)
        ax.set_yticklabels(z_lims)

    if i < 3:
        ax.set_xlabel(r'$\lambda$ (deg)', labelpad=-10)
        ax.set_xlim(lon_lims)
        ax.set_xticks(lon_lims)
        ax.set_xticklabels(lon_lims)
    else:
        ax.set_xlabel(r'$\vartheta$ (deg)', labelpad=-10)
        ax.set_xlim(lat_lims)
        ax.set_xticks(lat_lims)
        ax.set_xticklabels(lat_lims)

# Save figure ------------------------------------------------------------------
plt.suptitle(f't = {time:.1f} s', y=0.95)
fig.subplots_adjust(wspace=0.25, hspace=0.1)
plot_name = f'{plot_stem}_initial.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------- #
# FINAL PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(1, 2, figsize=(12, 6))
time_idx = -1
time = data_file['time'][time_idx]

for i, (ax, field_name, field_label, colour_scheme, contours,
        to_remove, slice_along) in \
        enumerate(zip(axarray.flatten(), final_field_names, final_field_labels,
                      final_colour_schemes, final_contours,
                      final_contours_to_remove, final_slice_along)):

    # Data extraction ----------------------------------------------------------
    if slice_along == 'z':
        field_full = extract_gusto_field(data_file, field_name, time_idx)
        coords_X_full, coords_Y_full, coords_Z_full = \
            extract_gusto_coords(data_file, field_name)

        # Reshape
        field_full, coords_X_full, coords_Y_full, _ = \
            reshape_gusto_data(field_full, coords_X_full,
                               coords_Y_full, coords_Z_full)

        # Take level for a horizontal slice
        field_data = field_full[:, level]
        # Abuse of names for coord variables but simplifies code below
        coords_X = coords_X_full[:, level]
        coords_Y = coords_Y_full[:, level]

    else:
        orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
            extract_gusto_vertical_slice(
                data_file, field_name, time_idx,
                slice_along=slice_along, slice_at=slice_at_lat
            )

        # Slices need regridding as points don't cleanly live along lon or lat = 0.0
        field_data, coords_X, coords_Y = \
            regrid_vertical_slice(
                plotting_grid[slice_along], slice_along, slice_at_lat,
                orig_coords_X, orig_coords_Y, orig_coords_Z, orig_field_data
            )
        # Scale coordinates
        coords_Y /= 1000.

    # Plot data ----------------------------------------------------------------
    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=to_remove)
    cf, lines = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(
        fig, cf, field_label, location='bottom', cbar_labelpad=-10,
        cbar_format=cbar_format[field_name]
    )
    if slice_along == 'z':
        tomplot_field_title(
            ax, r'$z = $ 5 km', minmax=True, field_data=field_data
        )
    elif slice_along == 'lat':
        tomplot_field_title(
            ax, r'$\vartheta = $ 0 deg', minmax=True, field_data=field_data
        )

    # Labels -------------------------------------------------------------------
    if i == 0:
        ax.set_xlabel(r'$\lambda$ (deg)', labelpad=-10)
        ax.set_xlim(lon_lims)
        ax.set_xticks(lon_lims)
        ax.set_xticklabels(lon_lims)
        ax.set_ylabel(r'$\vartheta$ (deg)', labelpad=-20)
        ax.set_ylim(lat_lims)
        ax.set_yticks(lat_lims)
        ax.set_yticklabels(lat_lims)
    else:
        ax.set_xlabel(r'$\lambda$ (deg)', labelpad=-10)
        ax.set_xlim(lon_lims)
        ax.set_xticks(lon_lims)
        ax.set_xticklabels(lon_lims)
        ax.set_ylabel(r'$z$ (km)', labelpad=-20)
        ax.set_ylim(z_lims)
        ax.set_yticks(z_lims)
        ax.set_yticklabels(z_lims)

# Save figure ------------------------------------------------------------------
plt.suptitle(f't = {time:.0f} s')
fig.subplots_adjust(wspace=0.18)
plot_name = f'{plot_stem}_final.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
