import numpy as np
import pdb
import sys
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
from tomplot import (extract_gusto_field, extract_gusto_coords,             
                        regrid_horizontal_slice)


def make_structured(filepath, field_name):

    # f1, dumpfreq = filepath.split('df')
    # f2, dt = f1.split('dt')

    results_dir = f'/data/home/sh1293/results/jupiter_sw/{filepath}'
    results_file_name = f'{results_dir}/field_output.nc'
    data_file = Dataset(results_file_name, 'r')
    times = np.array(data_file['time'])
    field_data = extract_gusto_field(data_file, field_name)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)

    tol = 1e-6
    unique_X = [coords_X[0]]
    unique_Y = [coords_Y[0]]
    for x in coords_X:
        if np.min(abs(x-unique_X)) > tol:
            unique_X.append(x)
    for y in coords_Y:
        if np.min(abs(y-unique_Y)) > tol:
            unique_Y.append(y)
    unique_X = np.array(np.sort(unique_X))
    unique_Y = np.array(np.sort(unique_Y))
    X, Y = np.meshgrid(unique_X, unique_Y)

    new_data = regrid_horizontal_slice(X, Y,
                                            coords_X, coords_Y, field_data)
    da_2d = xr.DataArray(data=new_data,
                    dims=['y', 'x', 'time'],
                    coords=dict(y=unique_Y, x=unique_X, time=times),
                    name=field_name)

    return da_2d, times