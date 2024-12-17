"""
Plots the linear thermal Galewsky jet test case.
 """
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, plot_field_quivers, tomplot_field_title,
    extract_gusto_coords, extract_gusto_field, regrid_horizontal_slice
)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file

results_file_name = f'{abspath(dirname(__file__))}/../../results/linear_thermal_galewsky/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/thermal_shallow_water/linear_thermal_galewsky'

# ---------------------------------------------------------------------------- #
# Initial plot details
# ---------------------------------------------------------------------------- #
init_field_names = ['u', 'D', 'RelativeVorticity', 'b']
init_colour_schemes = ['Oranges', 'YlGnBu', 'RdBu_r', 'PuRd_r']
init_field_labels = [r'$|u|$ (m s$^{-1}$)', r'$D$ (m)',
                     r'$\zeta$ (s$^{-1})$', r'$b$ (m s$^{-2}$)']
init_contours_to_remove = [None, None, 0.0, None]
init_contours = [np.linspace(0.0, 80.0, 9),
                 np.linspace(8900.0, 10200.0+1e-3, 12),
                 np.linspace(-2e-4, 2e-4, 17),
                 np.linspace(8.8, 9.8, 11)]

# ---------------------------------------------------------------------------- #
# Final plot details
# ---------------------------------------------------------------------------- #
final_field_names = ['RelativeVorticity', 'b']
final_colour_schemes = ['RdBu_r', 'PuRd_r']
final_field_labels = [r'$\zeta$ (s$^{-1}$)', r'$b$ (m s$^{-2}$)']
final_contours_to_remove = [0.0, None]
final_contours = [np.linspace(-2e-4, 2e-4, 17),
                  np.linspace(8.8, 9.8, 11)]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'tricontour'
xlims = [-180, 180]
ylims = [10, 80]

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# INITIAL PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(2, 2, figsize=(16, 12), sharex='all', sharey='all')
time_idx = 0

for i, (ax, field_name, colour_scheme, field_label, contour_to_remove, contours) in \
    enumerate(zip(
        axarray.flatten(), init_field_names, init_colour_schemes,
        init_field_labels, init_contours_to_remove, init_contours)):

    # Data extraction ----------------------------------------------------------
    if field_name == 'u':
        zonal_data = extract_gusto_field(data_file, 'u_zonal', time_idx=time_idx)
        meridional_data = extract_gusto_field(data_file, 'u_meridional', time_idx=time_idx)
        field_data = np.sqrt(zonal_data**2 + meridional_data**2)
        coords_X, coords_Y = extract_gusto_coords(data_file, 'u_zonal')

    else:
        field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
        coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
        time = data_file['time'][time_idx] / (24.*60.*60.)

    # Plot data ----------------------------------------------------------------
    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=contour_to_remove)
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)
    tomplot_field_title(ax, None, minmax=True, field_data=field_data)

    # Add quivers --------------------------------------------------------------
    if field_name == 'u':
        # Need to re-grid to lat-lon grid to get sensible looking quivers
        lon_1d = np.linspace(-180.0, 180.0, 91)
        lat_1d = np.linspace(-90.0, 90.0, 81)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')
        regrid_zonal_data = regrid_horizontal_slice(
            lon_2d, lat_2d, coords_X, coords_Y, zonal_data,
            periodic_fix='sphere'
        )
        regrid_meridional_data = regrid_horizontal_slice(
            lon_2d, lat_2d, coords_X, coords_Y, meridional_data,
            periodic_fix='sphere'
        )
        plot_field_quivers(
            ax, lon_2d, lat_2d, regrid_zonal_data, regrid_meridional_data,
            spatial_filter_step=(12, 1), magnitude_filter=1.0,
        )

    # Labels -------------------------------------------------------------------
    if i in [0, 2]:
        ax.set_ylabel(r'$\vartheta$ (deg)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    if i in [2, 3]:
        ax.set_xlabel(r'$\lambda$ (deg)', labelpad=-10)
        ax.set_xlim(xlims)
        ax.set_xticks(xlims)
        ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
fig.subplots_adjust(wspace=0.25)
plt.suptitle(f't = {time:.1f} days')
plot_name = f'{plot_stem}_initial.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------- #
# FINAL PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(1, 2, figsize=(16, 8), sharex='all', sharey='all')
time_idx = -1

for i, (ax, field_name, colour_scheme, field_label, contour_to_remove, contours) in \
    enumerate(zip(
        axarray, final_field_names, final_colour_schemes,
        final_field_labels, final_contours_to_remove, final_contours)):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx] / (24.*60.*60.)

    # Plot data ----------------------------------------------------------------
    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=contour_to_remove)
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)
    tomplot_field_title(ax, None, minmax=True, field_data=field_data)

    # Labels -------------------------------------------------------------------
    if i == 0:
        ax.set_ylabel(r'$\vartheta$ (deg)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    ax.set_xlabel(r'$\lambda$ (deg)', labelpad=-10)
    ax.set_xlim(xlims)
    ax.set_xticks(xlims)
    ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
plt.suptitle(f't = {time:.1f} days')
plot_name = f'{plot_stem}_final.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
