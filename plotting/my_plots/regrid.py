import numpy as np
import pdb
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
from tomplot import (set_tomplot_style, tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     regrid_vertical_slice, tomplot_field_title,
                     extract_gusto_vertical_slice, apply_gusto_domain,
                     reshape_gusto_data, extract_gusto_field,
                     extract_gusto_coords, area_restriction,
                     regrid_horizontal_slice, regrid_regular_horizontal_slice)


filepath = 'Relax_to_pole_and_CO2/annular_vortex_mars_55-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100sols_tracer_tophat-80_ref-4'

ref_lev = 4


def gaussian_lat_lon_grid(nlat, nlon):
    # Generate Gaussian latitudes
    x, _ = roots_legendre(nlat)  # Roots of Legendre polynomial (Gaussian quadrature points)
    
    # Convert Gaussian quadrature points (x) to latitude in degrees
    latitudes = np.arcsin(x) * (180.0 / np.pi)  # Convert from radians to degrees
    
    # Generate longitudes equally spaced between -180 and 180 degrees
    longitudes = np.linspace(-180, 179.9999, nlon, endpoint=True)
    
    return latitudes, longitudes

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file

results_dir = f'/data/home/sh1293/results/{filepath}'
# plot_dir = f'{results_dir}/plots'
results_file_name = f'{results_dir}/field_output.nc'
output_file_name = f'{results_dir}/regrid_output.nc'
# plot_name = f'{plot_dir}/eddy_enstrophy.pdf'
data_file = Dataset(results_file_name, 'r')

times = np.array(data_file['time'])

# lats, lons = gaussian_lat_lon_grid(40, 80)
# lats = np.arange(-90, 91, 1.5*(0.5)**(ref_lev-4))
lats = np.linspace(-90, 90, 120*2**(ref_lev-4)+1)
# lons = np.arange(-180, 181, 3*(0.5)**(ref_lev-4))
lons = np.linspace(-180, 181, 120*2**(ref_lev-4)+1)
X, Y = np.meshgrid(lons, lats)

ds_list=[]
for i in range(0, len(times)):
# for i in [0, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    print(i)
    for field_name in ['D', 'D_minus_H_rel_flag_less', 'u_meridional', 'u_zonal', 'PotentialVorticity', 'tracer', 'CO2cond_flag', 'CO2cond_flag_cumulative']:
        try:
            field_data = extract_gusto_field(data_file, field_name, time_idx=i)
        except:
            print(f'field {field_name} not present')
            continue
        # times = np.arange(np.shape(field_data)[1])
        coords_X, coords_Y = extract_gusto_coords(data_file, field_name)


        # field_data_t = field_data[:,i]

        # times = np.arange(np.shape(field_data)[1])
        # pdb.set_trace()
        if field_name == 'D_minus_H_rel_flag_less':
            new_data = regrid_horizontal_slice(X, Y,
                                                coords_X, coords_Y, field_data,
                                                periodic_fix='sphere')#, method='nearest')
        else:
            new_data = regrid_horizontal_slice(X, Y,
                                                coords_X, coords_Y, field_data,
                                                periodic_fix='sphere')
        da_2d = xr.DataArray(data=new_data.astype('float32'),
                        dims=['lat', 'lon'],
                        coords=dict(lat=lats.astype('float32'), lon=lons.astype('float32')),
                        name=field_name)
        da = da_2d.expand_dims(time=[times[i].astype('float32')])
        ds1 = da.to_dataset()
        if field_name == 'D' and i == 0:
            ds = ds1
        else:
            ds = xr.merge([ds, ds1])
#         ds_list.append(ds1)

# ds = xr.concat(ds_list, dim='time')

ds.to_netcdf(output_file_name)

print(f'Regridding done for \n {filepath}')