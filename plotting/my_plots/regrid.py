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
results_dir = f'/data/home/sh1293/firedrake-real-opt_may24/src/gusto/examples/shallow_water/results/Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_alpha--1_working_long'
plot_dir = f'{results_dir}/plots'
results_file_name = f'{results_dir}/field_output.nc'
output_file_name = f'{results_dir}/regrid_output.nc'
plot_name = f'{plot_dir}/eddy_enstrophy.pdf'
data_file = Dataset(results_file_name, 'r')
for field_name in ['D', 'D_minus_H_rel_flag_less', 'u_meridional', 'u_zonal', 'PotentialVorticity']:
    field_data = extract_gusto_field(data_file, field_name)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    times = np.arange(np.shape(field_data)[1])
    # lats = np.arange(-90, 91, 3)
    # lons = np.arange(-180, 181, 3)
    lats, lons = gaussian_lat_lon_grid(40, 80)
    X, Y = np.meshgrid(lons, lats)
    # pdb.set_trace()
    new_data = regrid_horizontal_slice(X, Y,
                                        coords_X, coords_Y, field_data)
    da = xr.DataArray(data=new_data.astype('float32'),
                    dims=['lat', 'lon', 'time'],
                    coords=dict(lat=lats.astype('float32'), lon=lons.astype('float32'), time=times.astype('float32')),
                    name=field_name)
    ds1 = da.to_dataset()
    if field_name == 'D':
        ds = ds1
    else:
        ds = xr.merge([ds, ds1])

ds.to_netcdf(output_file_name)