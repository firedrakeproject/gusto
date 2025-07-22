import numpy as np
import pdb
import sys
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


if len(sys.argv) != 3:
    print('Wrong number of arguments')
    sys.exit(1)

# filepath = 'Relax_to_pole_and_CO2/annular_vortex_mars_60-70_tau_r--2sol_tau_c--0.01sol_beta--1-0_A0-0-norel_len-100-300sols_tracer_strip-20-40_ref-4_pvmax-1-8'

# ref_lev = 4

filepath = sys.argv[1]
ref_lev = int(sys.argv[2])

print(f'inputs to regrid.py are filepath={filepath} and ref_lev={ref_lev}')

dumpfreq = 10

start_cut = True
start_frac = 1/3

dt = (0.5)**(ref_lev-4) * 450. * dumpfreq

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
# times = times[0:11838]
# times = np.hstack([[0.], times[11838:]])

# lats, lons = gaussian_lat_lon_grid(40, 80)
# lats = np.arange(-90, 91, 1.5*(0.5)**(ref_lev-4))
lats = np.linspace(-90, 90, 120*2**(ref_lev-4)+1)
# lons = np.arange(-180, 181, 3*(0.5)**(ref_lev-4))
lons = np.linspace(-180, 180, 120*2**(ref_lev-4)+1)
X, Y = np.meshgrid(lons, lats)

if ref_lev == 5:
    dt*=2
    lats1 = np.where(lats>=0, lats, np.nan)
    lats = lats1[~np.isnan(lats1)]
    X, Y = np.meshgrid(lons, lats)

ds_list=[]

if not start_cut:
    i_list = range(0, len(times))
if start_cut:
    i_range = range(int(len(times)*start_frac)-2, len(times))
    i_list = [0]
    i_list += (i_range)

# for i in range(0, len(times)):

for i in i_list:
# for i in [0, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984]:
    print(i)

    if data_file.variables['time'][i].item()%dt==0:
        print(data_file.variables['time'][i].item())

        # for field_name in ['D', 'D_minus_H_rel_flag_less', 'u_meridional', 'u_zonal', 'PotentialVorticity', 'tracer', 'CO2cond_flag', 'CO2cond_flag_cumulative', 'tracer_rs']:
        for field_name in ['D', 'PotentialVorticity', 'tracer', 'CO2cond_flag', 'CO2cond_flag_cumulative', 'tracer_rs']:
            try:
                field_data = extract_gusto_field(data_file, field_name, time_idx=i)
            except:
                print(f'field {field_name} not present')
                continue
            # times = np.arange(np.shape(field_data)[1])
            coords_X, coords_Y = extract_gusto_coords(data_file, field_name)

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


ds.to_netcdf(output_file_name)

print(f'Regridding done for \n {filepath}')