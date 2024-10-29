import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import Dataset
from tomplot import (set_tomplot_style, tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     regrid_vertical_slice, tomplot_field_title,
                     extract_gusto_vertical_slice, apply_gusto_domain,
                     reshape_gusto_data, extract_gusto_field,
                     extract_gusto_coords, area_restriction,
                     regrid_horizontal_slice)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
results_dir = f'/data/home/sh1293/results/Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_alpha--1_working_long'
plot_dir = f'{results_dir}/plots'
results_file_name = f'{results_dir}/field_output.nc'
plot_name = f'{plot_dir}/hovmoller.png'
data_file = Dataset(results_file_name, 'r')
field_name = 'PotentialVorticity'

field_data = extract_gusto_field(data_file, field_name)
coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
times = np.arange(np.shape(field_data)[1])
co2_data = extract_gusto_field(data_file, 'D_minus_H_rel_flag_less')
co2_X, co2_Y = extract_gusto_coords(data_file, 'D_minus_H_rel_flag_less')
# limits = {'Y': (64, 65)}
# cut_field_data, cut_coords_X, cut_coords_Y = area_restriction(field_data, coords_X, coords_Y, limits)
lats = 70
lons = np.arange(-180, 180.1, 0.1)
new_data = regrid_horizontal_slice(coords_X, lats,
                                    coords_X, coords_Y, field_data)
new_co2 = regrid_horizontal_slice(lons, lats,
                                    co2_X, co2_Y, co2_data)

fig, ax = plt.subplots(1, 1, figsize=(10,10))
time_early = -1600
time_late = -1000
plot_times = times[time_early:time_late]
plot_data = np.transpose(new_data)[time_early:time_late,:]
plot_co2 = np.transpose(new_co2)[time_early:time_late,:]
co2_mask = np.where(plot_co2 == 0, np.nan, 1)
cf = ax.contourf(coords_X, plot_times, plot_data, cmap='OrRd', levels=21)
# cf1 = ax.contour(lons, plot_times, co2_mask, colors='black')
cf1 = ax.contourf(lons, plot_times, co2_mask, colors='none', levels=[0, 1.5], hatches=['xx'])
add_colorbar_ax(ax, cf, cbar_label='PotentialVorticity', location='right', shrink=0.5)
# add_colorbar_ax(ax, cf1, cbar_label='co2_mask', location='left')
# artists, labels = cf1.legend_elements(str_format='{:2.1f}'.format)
# ax.legend(artists, labels, handleheight=2, framealpha=1)
ax.set_xlabel('longitude')
ax.set_ylabel('time')
plt.savefig(plot_name)